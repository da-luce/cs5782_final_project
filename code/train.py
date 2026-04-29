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

# Dataset-specific configuration for SST-2 and MNLI
DATASET_CONFIG = {
    "sst2": {
        "num_labels": 2,
        "text_fields": ["sentence"],
        "val_split": "validation",
    },
    "mnli": {
        "num_labels": 3,
        "text_fields": ["premise", "hypothesis"],
        "val_split": "validation_matched",
    },
}


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def run_experiment(mode, dataset_name, train_samples=TRAIN_SAMPLES, val_samples=VAL_SAMPLES, epochs=EPOCHS):
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Choose from: {list(DATASET_CONFIG.keys())}")

    cfg = DATASET_CONFIG[dataset_name]
    num_labels = cfg["num_labels"]
    text_fields = cfg["text_fields"]
    val_split = cfg["val_split"]

    device = get_device()
    print(f"Using device: {device}")

    start_time = time.time()

    dataset = load_dataset("glue", dataset_name)

    # Select a subset for faster training
    dataset["train"] = dataset["train"].select(range(train_samples))
    dataset[val_split] = dataset[val_split].select(range(val_samples))

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    def tokenize(batch):
        return tokenizer(*[batch[f] for f in text_fields], truncation=True, padding="max_length", max_length=128)

    # Tokenize after selecting the subset to avoid processing the full dataset
    dataset = dataset.map(tokenize, batched=True)

    model = get_baseline_model(num_labels=num_labels) if mode == "baseline" else get_lora_model(num_labels=num_labels)

    param_info = count_parameters(model)
    print(f"\nParameter counts ({mode}):")
    print(f"  Total:     {param_info['total']:,}")
    print(f"  Trainable: {param_info['trainable']:,} ({param_info['trainable_pct']}%)")
    print(f"  Frozen:    {param_info['frozen']:,}\n")

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=labels)

    args = TrainingArguments(
        output_dir=f"results/{mode}_{dataset_name}",
        eval_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-5 if mode == "baseline" else 2e-4,
        per_device_train_batch_size=8,
        num_train_epochs=epochs,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset[val_split],
        compute_metrics=compute_metrics,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    trainer.train()

    eval_result = trainer.evaluate()

    model_dir = f"./models/{mode}_{dataset_name}"
    trainer.save_model(model_dir)

    end_time = time.time()

    def get_dir_size_mb(path):
        total_size = 0
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
        return total_size / (1024 * 1024)

    checkpoint_size_mb = get_dir_size_mb(model_dir)
    peak_memory_mb = 0
    if torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    info = {
        "mode": mode,
        "dataset": dataset_name,
        "device": device,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "epochs": epochs,
        "learning_rate": 2e-5 if mode == "baseline" else 2e-4,
        "batch_size": 8,
        "eval_results": eval_result,
        "training_time_sec": end_time - start_time,
        "trainable_params": param_info,
        "model_output_dir": model_dir,
        "checkpoint_size_mb": checkpoint_size_mb,
        "peak_memory_mb": peak_memory_mb,
        "log_history": trainer.state.log_history,
    }

    os.makedirs("./results", exist_ok=True)

    info_path = f"./results/{mode}_{dataset_name}.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)

    print(f"\nSaved model to ./models/{mode}_{dataset_name}")
    print(f"Saved results to {info_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["baseline", "lora"])
    parser.add_argument("--dataset", type=str, default="sst2", choices=list(DATASET_CONFIG.keys()))
    parser.add_argument("--train_samples", type=int, default=TRAIN_SAMPLES)
    parser.add_argument("--val_samples", type=int, default=VAL_SAMPLES)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    run_experiment(args.mode, args.dataset, args.train_samples, args.val_samples, args.epochs)


if __name__ == "__main__":
    main()
