import argparse
import json
import os
import time

from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer
import numpy as np
import evaluate
import torch

from models import get_baseline_model, get_lora_model
from lora import count_parameters

TRAIN_SAMPLES = 5000
VAL_SAMPLES = 500
EPOCHS = 5
LORA_RANK = 8

DATASET_CONFIG = {
    # sst2: binary sentiment classification
    "sst2": {
        "num_labels": 2,
        "text_fields": ["sentence"],
        "val_split": "validation",
    },
    # mnli: 3-way NLI classification
    "mnli": {
        "num_labels": 3,
        "text_fields": ["premise", "hypothesis"],
        "val_split": "validation_matched",
    },
}

MODE_CONFIG = {
    # baseline: no fine-tuning, just evaluation with pretrained weights
    "baseline": {
        "is_training": False,
        "learning_rate": 0.0,
        "get_model": get_baseline_model, 
    },
    # finetune: update all model parameters
    "finetune": {
        "is_training": True,
        "learning_rate": 2e-5,
        "get_model": get_baseline_model,
    },
    # lora: update only LoRA parameters
    "lora": {
        "is_training": True,
        "learning_rate": 2e-4,
        "get_model": get_lora_model,
    }
}


def get_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


def run_experiment(mode, dataset_name, train_samples=TRAIN_SAMPLES, val_samples=VAL_SAMPLES, epochs=EPOCHS, rank=LORA_RANK):
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Choose from: {list(DATASET_CONFIG.keys())}")
    if mode not in MODE_CONFIG:
        raise ValueError(f"Unsupported mode '{mode}'. Choose from: {list(MODE_CONFIG.keys())}")

    # 1. Load Configurations
    data_cfg = DATASET_CONFIG[dataset_name]
    mode_cfg = MODE_CONFIG[mode]
    is_training = mode_cfg["is_training"]

    device = get_device()
    print(f"Using device: {device}")

    # 2. Data Loading & Subsetting
    dataset = load_dataset("glue", dataset_name)
    val_split = dataset[data_cfg["val_split"]]
    if val_samples > len(val_split):
        print(f"Warning: val_samples={val_samples} exceeds dataset size {len(val_split)}, using {len(val_split)}")
        val_samples = len(val_split)
    val_dataset = val_split.select(range(val_samples))
    if is_training:
        train_split = dataset["train"]
        if train_samples > len(train_split):
            print(f"Warning: train_samples={train_samples} exceeds dataset size {len(train_split)}, using {len(train_split)}")
            train_samples = len(train_split)
        train_dataset = train_split.select(range(train_samples))
    else:
        train_dataset = None

    # 3. Tokenization 
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    def tokenize(batch):
        return tokenizer(*[batch[f] for f in data_cfg["text_fields"]], truncation=True, padding="max_length", max_length=128)

    val_dataset = val_dataset.map(tokenize, batched=True)
    if is_training:
        train_dataset = train_dataset.map(tokenize, batched=True)

    # 4. Model Setup
    if mode == "lora":
        model = mode_cfg["get_model"](num_labels=data_cfg["num_labels"], r=rank)
    else:
        model = mode_cfg["get_model"](num_labels=data_cfg["num_labels"])
    param_info = count_parameters(model)
    
    print(f"\nParameter counts ({mode}):")
    print(f"  Total:     {param_info['total']:,}")
    print(f"  Trainable: {param_info['trainable']:,} ({param_info['trainable_pct']}%)")
    print(f"  Frozen:    {param_info['frozen']:,}\n")

    # 5. Metrics & Trainer Setup
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=labels)

    batch_size = 8

    run_tag = f"{mode}_{dataset_name}_r{rank}" if mode == "lora" else f"{mode}_{dataset_name}"

    args = TrainingArguments(
        output_dir=f"results/{run_tag}",
        eval_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=mode_cfg["learning_rate"],
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs if is_training else 0,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # 6. Execution (Train & Eval with independent timing)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    train_time_sec = 0.0
    model_dir = None
    if is_training:
        train_start = time.time()
        trainer.train()
        train_time_sec = time.time() - train_start

        model_dir = f"./models/{run_tag}"
        trainer.save_model(model_dir)
        print(f"\nSaved model to {model_dir}")

    eval_start = time.time()
    eval_result = trainer.evaluate()
    eval_time_sec = time.time() - eval_start

    # Peak VRAM usage
    if device == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
    elif device == "mps":
        peak_memory_mb = torch.mps.current_allocated_memory() / 1024**2
    else:
        peak_memory_mb = 0.0

    # Saved checkpoint size on disk
    checkpoint_size_mb = 0.0
    if model_dir and os.path.isdir(model_dir):
        for dirpath, _, filenames in os.walk(model_dir):
            for fname in filenames:
                checkpoint_size_mb += os.path.getsize(os.path.join(dirpath, fname))
        checkpoint_size_mb /= 1024**2

    # 7. Structured Logging & Saving Results
    metadata = {"mode": mode, "dataset": dataset_name, "device": device}
    if mode == "lora":
        metadata["r"] = rank

    info = {
        "metadata": metadata,
        "model": param_info,
        "training": {
            "executed": is_training,
            "samples": train_samples if is_training else 0,
            "epochs": epochs if is_training else 0,
            "learning_rate": mode_cfg["learning_rate"],
            "batch_size": batch_size,
            "time_sec": train_time_sec,
            "peak_memory_mb": peak_memory_mb,
            "checkpoint_size_mb": checkpoint_size_mb,
            "log_history": trainer.state.log_history,
        },
        "eval": {
            "samples": val_samples,
            "time_sec": eval_time_sec,
            "results": eval_result,
        }
    }

    if model_dir:
        info["training"]["model_output_dir"] = model_dir

    os.makedirs("./results", exist_ok=True)
    info_path = f"./results/{run_tag}.json"
    
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)

    print(f"Saved results to {info_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=list(MODE_CONFIG.keys()))
    parser.add_argument("--dataset", type=str, default="sst2", choices=list(DATASET_CONFIG.keys()))
    parser.add_argument("--train_samples", type=int, default=TRAIN_SAMPLES)
    parser.add_argument("--val_samples", type=int, default=VAL_SAMPLES)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--rank", type=int, default=LORA_RANK,
                        help="LoRA rank (only used when --mode lora, default 8)")

    args = parser.parse_args()
    run_experiment(args.mode, args.dataset, args.train_samples, args.val_samples, args.epochs, args.rank)


if __name__ == "__main__":
    main()