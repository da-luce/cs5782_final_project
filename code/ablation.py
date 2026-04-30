"""
LoRA rank ablation experiment.

Run a single rank:
    python ablation.py --rank 8 --dataset sst2

Run all ranks individually (e.g. from a notebook, one cell per rank):
    python ablation.py --rank 2  --dataset sst2
    python ablation.py --rank 4  --dataset sst2
    python ablation.py --rank 8  --dataset sst2
    python ablation.py --rank 16 --dataset sst2

Compare all saved rank results for a dataset:
    python ablation.py --compare --dataset sst2

Each run saves to results/ablation_r{rank}_{dataset}.json so runs are
independent and any single rank can be rerun without losing the others.

Standalone — does not import from train.py or models.py.
"""

import argparse
import glob
import json
import os
import time

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from lora import apply_lora, count_parameters

# ---------------------------------------------------------------------------
# Config
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
    val_split = cfg["val_split"]

    dataset = load_dataset("glue", dataset_name)
    dataset["train"]   = dataset["train"].select(range(train_samples))
    dataset[val_split] = dataset[val_split].select(range(val_samples))

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    def tokenize(batch):
        return tokenizer(
            *[batch[f] for f in cfg["text_fields"]],
            truncation=True,
            padding="max_length",
            max_length=128,
        )

    return dataset.map(tokenize, batched=True), val_split


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


def result_path(rank, dataset_name):
    return f"./results/ablation_r{rank}_{dataset_name}.json"


# ---------------------------------------------------------------------------
# Single-rank experiment
# ---------------------------------------------------------------------------

def run_rank(rank, dataset_name, train_samples, val_samples):
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Choose from: {list(DATASET_CONFIG.keys())}")

    cfg    = DATASET_CONFIG[dataset_name]
    device = get_device()

    print(f"Rank ablation | r={rank} | dataset={dataset_name} | device={device}")
    print(f"train_samples={train_samples} | val_samples={val_samples}\n")

    dataset, val_split = load_and_tokenize(dataset_name, train_samples, val_samples)

    model      = build_lora_model(cfg["num_labels"], rank)
    param_info = count_parameters(model)
    print(f"Trainable: {param_info['trainable']:,} ({param_info['trainable_pct']}%)\n")

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        return metric.compute(predictions=np.argmax(logits, axis=-1), references=labels)

    t0 = time.time()

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=f"results/ablation_r{rank}_{dataset_name}",
            eval_strategy="epoch",
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=BATCH_SIZE,
            num_train_epochs=NUM_EPOCHS,
            report_to="none",
        ),
        train_dataset=dataset["train"],
        eval_dataset=dataset[val_split],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_result = trainer.evaluate()
    elapsed     = time.time() - t0

    result = {
        "r":                 rank,
        "dataset":           dataset_name,
        "device":            device,
        "lora_alpha":        LORA_ALPHA,
        "scaling":           LORA_ALPHA / rank,
        "train_samples":     train_samples,
        "val_samples":       val_samples,
        "epochs":            NUM_EPOCHS,
        "batch_size":        BATCH_SIZE,
        "learning_rate":     LEARNING_RATE,
        "trainable_params":  param_info,
        "eval_accuracy":     eval_result.get("eval_accuracy"),
        "eval_results":      eval_result,
        "training_time_sec": elapsed,
    }

    os.makedirs("./results", exist_ok=True)
    with open(result_path(rank, dataset_name), "w") as f:
        json.dump(result, f, indent=4)

    print(f"\nr={rank} | accuracy={result['eval_accuracy']:.4f} | "
          f"trainable={param_info['trainable']:,} | time={elapsed/60:.1f}m")
    print(f"Saved to {result_path(rank, dataset_name)}")


# ---------------------------------------------------------------------------
# Comparison table (loads all saved rank results for a dataset)
# ---------------------------------------------------------------------------

def compare_ranks(dataset_name):
    pattern = f"./results/ablation_r*_{dataset_name}.json"
    files   = sorted(glob.glob(pattern), key=lambda p: int(p.split("_r")[1].split("_")[0]))

    if not files:
        print(f"No ablation results found matching {pattern}")
        print("Run individual ranks first: python ablation.py --rank <r> --dataset <dataset>")
        return

    print(f"Rank ablation comparison — dataset: {dataset_name}\n")
    print(f"{'r':>4}  {'Trainable':>12}  {'% of total':>10}  {'Accuracy':>10}  {'Time (min)':>12}")
    print("─" * 56)

    for fpath in files:
        with open(fpath) as f:
            d = json.load(f)
        p = d["trainable_params"]
        print(f"{d['r']:>4}  {p['trainable']:>12,}  {p['trainable_pct']:>9.4f}%  "
              f"{d['eval_accuracy']:>10.4f}  {d['training_time_sec']/60:>11.1f}m")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LoRA rank ablation")
    parser.add_argument("--dataset",       type=str, default="sst2",
                        choices=list(DATASET_CONFIG.keys()))
    parser.add_argument("--rank",          type=int, default=None,
                        help="Single LoRA rank to train, e.g. --rank 8")
    parser.add_argument("--compare",       action="store_true",
                        help="Print comparison table from all saved rank results")
    parser.add_argument("--train_samples", type=int, default=DEFAULT_TRAIN_SAMPLES)
    parser.add_argument("--val_samples",   type=int, default=DEFAULT_VAL_SAMPLES)
    args = parser.parse_args()

    if args.compare:
        compare_ranks(args.dataset)
    elif args.rank is not None:
        run_rank(args.rank, args.dataset, args.train_samples, args.val_samples)
    else:
        parser.error("Provide --rank <r> to train or --compare to print results.")


if __name__ == "__main__":
    main()
