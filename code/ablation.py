"""
Rank ablation experiment: trains a LoRA model for each rank in --ranks and
saves raw results to results/ablation_{dataset}.json.

Standalone — does not import from train.py or models.py so it can be
developed independently without merge conflicts.

Usage:
    python ablation.py --dataset sst2 --ranks 2 4 8 16
    python ablation.py --dataset mnli  --ranks 2 4 8 16 --train_samples 5000
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import evaluate
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from lora import apply_lora, count_parameters

# ---------------------------------------------------------------------------
# Dataset config (mirrors train.py — kept here to stay self-contained)
# ---------------------------------------------------------------------------

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

DEFAULT_RANKS        = [2, 4, 8, 16]
DEFAULT_TRAIN_SAMPLES = 2000
DEFAULT_VAL_SAMPLES   = 500
LORA_ALPHA            = 16.0
LORA_DROPOUT          = 0.1
LEARNING_RATE         = 2e-4
BATCH_SIZE            = 8
NUM_EPOCHS            = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_and_tokenize(dataset_name, train_samples, val_samples):
    cfg = DATASET_CONFIG[dataset_name]
    text_fields = cfg["text_fields"]
    val_split   = cfg["val_split"]

    dataset = load_dataset("glue", dataset_name)
    dataset["train"]    = dataset["train"].select(range(train_samples))
    dataset[val_split]  = dataset[val_split].select(range(val_samples))

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    def tokenize(batch):
        return tokenizer(
            *[batch[f] for f in text_fields],
            truncation=True,
            padding="max_length",
            max_length=128,
        )

    dataset = dataset.map(tokenize, batched=True)
    return dataset, val_split


def build_lora_model(num_labels, r):
    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base", num_labels=num_labels
    )
    return apply_lora(
        model,
        target_modules=["query", "value"],
        r=r,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        modules_to_save=["classifier"],
    )


# ---------------------------------------------------------------------------
# Main ablation loop
# ---------------------------------------------------------------------------

def run_ablation(dataset_name, ranks, train_samples, val_samples):
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Choose from: {list(DATASET_CONFIG.keys())}")

    cfg        = DATASET_CONFIG[dataset_name]
    num_labels = cfg["num_labels"]
    device     = get_device()

    print(f"Rank ablation | dataset={dataset_name} | ranks={ranks} | device={device}")
    print(f"train_samples={train_samples} | val_samples={val_samples}\n")

    dataset, val_split = load_and_tokenize(dataset_name, train_samples, val_samples)

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=labels)

    results = []

    for r in ranks:
        print(f"{'─'*50}")
        print(f"Training LoRA with r={r} ...")

        model      = build_lora_model(num_labels, r)
        param_info = count_parameters(model)

        print(f"  Trainable: {param_info['trainable']:,} ({param_info['trainable_pct']}%)")

        t0 = time.time()

        training_args = TrainingArguments(
            output_dir=f"results/ablation_r{r}_{dataset_name}",
            eval_strategy="epoch",
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=BATCH_SIZE,
            num_train_epochs=NUM_EPOCHS,
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset[val_split],
            compute_metrics=compute_metrics,
        )

        trainer.train()
        eval_result = trainer.evaluate()
        elapsed     = time.time() - t0

        entry = {
            "r":                   r,
            "lora_alpha":          LORA_ALPHA,
            "scaling":             LORA_ALPHA / r,
            "trainable_params":    param_info,
            "eval_accuracy":       eval_result.get("eval_accuracy"),
            "eval_results":        eval_result,
            "training_time_sec":   elapsed,
        }
        results.append(entry)

        print(f"  r={r:>2} | accuracy={entry['eval_accuracy']:.4f} | "
              f"trainable={param_info['trainable']:,} | time={elapsed/60:.1f}m")

    summary = {
        "dataset":       dataset_name,
        "device":        device,
        "train_samples": train_samples,
        "val_samples":   val_samples,
        "epochs":        NUM_EPOCHS,
        "batch_size":    BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "ranks_tested":  ranks,
        "results":       results,
    }

    os.makedirs("./results", exist_ok=True)
    out_path = f"./results/ablation_{dataset_name}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\n{'═'*50}")
    print(f"Ablation complete. Results saved to {out_path}")
    print(f"\n{'r':>4}  {'Trainable':>12}  {'% of total':>10}  {'Accuracy':>10}")
    print(f"{'─'*42}")
    for entry in results:
        p = entry["trainable_params"]
        print(f"{entry['r']:>4}  {p['trainable']:>12,}  {p['trainable_pct']:>9.4f}%  {entry['eval_accuracy']:>10.4f}")


def main():
    parser = argparse.ArgumentParser(description="LoRA rank ablation experiment")
    parser.add_argument("--dataset",       type=str, default="sst2",
                        choices=list(DATASET_CONFIG.keys()))
    parser.add_argument("--ranks",         type=int, nargs="+", default=DEFAULT_RANKS,
                        help="List of LoRA ranks to sweep, e.g. --ranks 2 4 8 16")
    parser.add_argument("--train_samples", type=int, default=DEFAULT_TRAIN_SAMPLES)
    parser.add_argument("--val_samples",   type=int, default=DEFAULT_VAL_SAMPLES)
    args = parser.parse_args()

    run_ablation(args.dataset, sorted(args.ranks), args.train_samples, args.val_samples)


if __name__ == "__main__":
    main()
