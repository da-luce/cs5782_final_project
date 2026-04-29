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


def run_experiment(mode, dataset_name, train_samples=TRAIN_SAMPLES, val_samples=VAL_SAMPLES, epochs=EPOCHS):
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
    val_dataset = dataset[data_cfg["val_split"]].select(range(val_samples))
    train_dataset = dataset["train"].select(range(train_samples)) if is_training else None

    # 3. Tokenization 
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    def tokenize(batch):
        return tokenizer(*[batch[f] for f in data_cfg["text_fields"]], truncation=True, padding="max_length", max_length=128)

    val_dataset = val_dataset.map(tokenize, batched=True)
    if is_training:
        train_dataset = train_dataset.map(tokenize, batched=True)

    # 4. Model Setup
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

    args = TrainingArguments(
        output_dir=f"results/{mode}_{dataset_name}",
        eval_strategy="epoch",
        learning_rate=mode_cfg["learning_rate"],
        per_device_train_batch_size=8,
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
    train_time_sec = 0.0
    if is_training:
        train_start = time.time()
        trainer.train()
        train_time_sec = time.time() - train_start
        
        model_dir = f"./models/{mode}_{dataset_name}"
        trainer.save_model(model_dir)
        print(f"\nSaved model to {model_dir}")

    eval_start = time.time()
    eval_result = trainer.evaluate()
    eval_time_sec = time.time() - eval_start

    # 7. Structured Logging & Saving Results
    info = {
        "metadata": {
            "mode": mode,
            "dataset": dataset_name,
            "device": device,
        },
        "model": param_info,
        "training": {
            "executed": is_training,
            "samples": train_samples if is_training else 0,
            "epochs": epochs if is_training else 0,
            "learning_rate": mode_cfg["learning_rate"],
            "batch_size": 8,
            "time_sec": train_time_sec,
        },
        "eval": {
            "samples": val_samples,
            "time_sec": eval_time_sec,
            "results": eval_result,
        }
    }
    
    if is_training:
        info["training"]["model_output_dir"] = model_dir

    os.makedirs("./results", exist_ok=True)
    info_path = f"./results/{mode}_{dataset_name}.json"
    
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
    
    args = parser.parse_args()
    run_experiment(args.mode, args.dataset, args.train_samples, args.val_samples, args.epochs)


if __name__ == "__main__":
    main()