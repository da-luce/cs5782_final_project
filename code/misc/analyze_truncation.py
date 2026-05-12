"""
Quantify how many training/validation samples are truncated at max_length=128.

Tokenizes each split WITHOUT truncation to measure true sequence lengths,
then reports how many samples exceed 128 tokens (i.e., are truncated in training).
"""

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

MAX_LENGTH = 128

DATASET_CONFIG = {
    "sst2": {
        "text_fields": ["sentence"],
        "val_split": "validation",
        "train_samples": 50000,
        "val_samples": 500,
    },
    "mnli": {
        "text_fields": ["premise", "hypothesis"],
        "val_split": "validation_matched",
        "train_samples": 50000,
        "val_samples": 500,
    },
}

tokenizer = AutoTokenizer.from_pretrained("roberta-base")


def get_lengths(dataset, text_fields):
    lengths = []
    for example in dataset:
        tokens = tokenizer(*[example[f] for f in text_fields], truncation=False, padding=False)
        lengths.append(len(tokens["input_ids"]))
    return np.array(lengths)


def report(name, split_name, lengths):
    truncated = (lengths > MAX_LENGTH).sum()
    total = len(lengths)
    pct = 100.0 * truncated / total
    print(f"  {name} [{split_name}]: {truncated}/{total} truncated ({pct:.1f}%)")
    print(f"    length stats — min: {lengths.min()}, median: {int(np.median(lengths))}, "
          f"p95: {int(np.percentile(lengths, 95))}, max: {lengths.max()}")


print(f"Analyzing truncation at max_length={MAX_LENGTH}\n")

for dataset_name, cfg in DATASET_CONFIG.items():
    print(f"=== {dataset_name.upper()} ===")
    raw = load_dataset("glue", dataset_name)

    train_split = raw["train"]
    n_train = min(cfg["train_samples"], len(train_split))
    train_data = train_split.select(range(n_train))
    train_lengths = get_lengths(train_data, cfg["text_fields"])
    report(dataset_name, f"train[:{ n_train}]", train_lengths)

    val_split = raw[cfg["val_split"]]
    n_val = min(cfg["val_samples"], len(val_split))
    val_data = val_split.select(range(n_val))
    val_lengths = get_lengths(val_data, cfg["text_fields"])
    report(dataset_name, f"val[:{n_val}]", val_lengths)

    print()
