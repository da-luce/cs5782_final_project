"""Training driver for the LoRA re-implementation experiments.

Supports both:
    --mode baseline    full fine-tuning of RoBERTa-base
    --mode lora        backbone frozen, custom LoRA on attention W_q, W_v

Datasets:
    --dataset sst2     GLUE SST-2 (binary sentiment, single sentence)
    --dataset mnli     GLUE MNLI (three-way NLI, sentence pair)

Defaults are aligned with Table 9 (RoBERTa-base, LoRA) and standard RoBERTa-base
fine-tuning recipes for the baseline. Pass any flag to override.
"""

import argparse
import json
import os
import time

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from lora import count_parameters, format_parameter_report
from models import get_baseline_model, get_lora_model


DATASET_CONFIG = {
    "sst2": {
        "num_labels": 2,
        "text_fields": ["sentence"],
        "validation_keys": ["validation"],
        "metric_name": "accuracy",
    },
    "mnli": {
        "num_labels": 3,
        "text_fields": ["premise", "hypothesis"],
        "validation_keys": ["validation_matched", "validation_mismatched"],
        "metric_name": "accuracy",
    },
}

# (mode, dataset) -> dict of recommended hyperparameters.
#
# These defaults are Colab-friendly: same epochs / batch across modes for an
# apples-to-apples comparison, with mode-specific LRs (Table 9 of the paper for
# LoRA, standard RoBERTa-base FT for the baseline). Combined with the default
# --train_samples below, each run finishes in ~10-20 minutes on a T4.
#
# Paper-scale settings (arXiv:2106.09685, Table 9 — for the report):
#     baseline  : full data, 3 epochs,  batch 32, lr 2e-5
#     LoRA SST-2: full data, 60 epochs, batch 16, lr 5e-4
#     LoRA MNLI : full data, 30 epochs, batch 16, lr 5e-4
# To reproduce paper-scale, pass: --train_samples -1 --epochs N --batch_size 16
RECOMMENDED_HPARAMS = {
    ("baseline", "sst2"): {"lr": 2e-5, "epochs": 5, "batch_size": 32},
    ("baseline", "mnli"): {"lr": 2e-5, "epochs": 5, "batch_size": 32},
    ("lora", "sst2"): {"lr": 5e-4, "epochs": 5, "batch_size": 32},
    ("lora", "mnli"): {"lr": 5e-4, "epochs": 5, "batch_size": 32},
}


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_tokenize_fn(tokenizer, text_fields, max_length):
    if len(text_fields) == 1:
        f = text_fields[0]

        def tokenize(batch):
            return tokenizer(batch[f], truncation=True, max_length=max_length)

    elif len(text_fields) == 2:
        f1, f2 = text_fields

        def tokenize(batch):
            return tokenizer(
                batch[f1], batch[f2], truncation=True, max_length=max_length
            )

    else:
        raise ValueError(f"Unsupported number of text fields: {text_fields}")

    return tokenize


def preprocess(ds, tokenize_fn, text_fields):
    ds = ds.map(tokenize_fn, batched=True, remove_columns=text_fields)
    if "idx" in ds.column_names:
        ds = ds.remove_columns("idx")
    if "label" in ds.column_names:
        ds = ds.rename_column("label", "labels")
    return ds


def maybe_subset(ds, n: int, seed: int):
    if n is None or n < 0 or n >= len(ds):
        return ds
    return ds.shuffle(seed=seed).select(range(n))


def run_experiment(args):
    set_seed(args.seed)

    cfg = DATASET_CONFIG[args.dataset]
    device = get_device()
    print(
        f"Mode: {args.mode} | Dataset: {args.dataset} | "
        f"Device: {device} | Seed: {args.seed}"
    )

    start_time = time.time()

    raw = load_dataset("glue", args.dataset)

    train_raw = maybe_subset(raw["train"], args.train_samples, args.seed)
    val_raws = {
        k: maybe_subset(raw[k], args.val_samples, args.seed)
        for k in cfg["validation_keys"]
    }

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    tokenize_fn = make_tokenize_fn(
        tokenizer, cfg["text_fields"], args.max_seq_length
    )

    train_ds = preprocess(train_raw, tokenize_fn, cfg["text_fields"])
    val_dss = {
        k: preprocess(v, tokenize_fn, cfg["text_fields"]) for k, v in val_raws.items()
    }

    if args.mode == "baseline":
        model = get_baseline_model(num_labels=cfg["num_labels"])
        replaced = []
    else:
        model, replaced = get_lora_model(
            num_labels=cfg["num_labels"],
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )

    param_report = count_parameters(model)
    print(format_parameter_report(param_report))
    if replaced:
        print(f"Injected LoRA into {len(replaced)} modules.")
        print(f"  e.g. {replaced[0]}")

    metric = evaluate.load(cfg["metric_name"])

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=labels)

    primary_eval_key = cfg["validation_keys"][0]
    primary_eval_ds = val_dss[primary_eval_key]

    output_dir = os.path.join(
        "results", "checkpoints", f"{args.mode}_{args.dataset}_seed{args.seed}"
    )
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(args.batch_size, 32),
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="linear",
        logging_steps=50,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=primary_eval_ds,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    trainer.train()

    eval_results = {}
    for key, ds in val_dss.items():
        result = trainer.evaluate(ds, metric_key_prefix=key)
        eval_results[key] = result
        print(f"[{key}] {result}")

    duration = time.time() - start_time

    info = {
        "mode": args.mode,
        "dataset": args.dataset,
        "seed": args.seed,
        "device": device,
        "train_size": len(train_ds),
        "validation_sizes": {k: len(v) for k, v in val_dss.items()},
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "max_seq_length": args.max_seq_length,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "lora": (
            {
                "r": args.lora_r,
                "alpha": args.lora_alpha,
                "dropout": args.lora_dropout,
                "n_injected_modules": len(replaced),
            }
            if args.mode == "lora"
            else None
        ),
        "param_report": param_report,
        "eval_results": eval_results,
        "training_time_sec": duration,
    }

    info_dir = os.path.join("results", "info")
    os.makedirs(info_dir, exist_ok=True)
    info_path = os.path.join(
        info_dir, f"{args.mode}_{args.dataset}_seed{args.seed}.json"
    )
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"Saved run info to {info_path}")

    if args.save_model:
        model_dir = os.path.join(
            "models", f"{args.mode}_{args.dataset}_seed{args.seed}"
        )
        os.makedirs(model_dir, exist_ok=True)
        trainer.save_model(model_dir)
        print(f"Saved model to {model_dir}")

    print(f"Total time: {duration:.1f}s")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument("--mode", type=str, required=True, choices=["baseline", "lora"])
    p.add_argument(
        "--dataset", type=str, default="sst2", choices=list(DATASET_CONFIG)
    )
    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate (defaults from RECOMMENDED_HPARAMS).",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override per-device train batch size.",
    )

    p.add_argument("--max_seq_length", type=int, default=128)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--weight_decay", type=float, default=0.0)

    p.add_argument(
        "--train_samples",
        type=int,
        default=10000,
        help="Subset training set to this many examples (-1 = full).",
    )
    p.add_argument(
        "--val_samples",
        type=int,
        default=-1,
        help="Subset each validation split to this many examples (-1 = full).",
    )

    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=float, default=8.0)
    p.add_argument("--lora_dropout", type=float, default=0.1)

    p.add_argument(
        "--save_model",
        action="store_true",
        help="Save the trained model to models/<mode>_<dataset>_seed<seed>/",
    )

    args = p.parse_args()

    rec = RECOMMENDED_HPARAMS[(args.mode, args.dataset)]
    if args.lr is None:
        args.lr = rec["lr"]
    if args.epochs is None:
        args.epochs = rec["epochs"]
    if args.batch_size is None:
        args.batch_size = rec["batch_size"]

    return args


def main():
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
