#!/usr/bin/env python3
"""Generate diagrams from training result JSON files."""

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np

MODE_ORDER = ["baseline", "lora", "finetune"]
MODE_LABELS = {"baseline": "Baseline", "lora": "LoRA", "finetune": "Full Fine-tune"}
MODE_COLORS = {"baseline": "#4878d0", "lora": "#ee854a", "finetune": "#6acc65"}


def load_results(results_dir, lora_rank=8):
    data = {}
    for path in glob.glob(os.path.join(results_dir, "*.json")):
        with open(path) as f:
            result = json.load(f)
        mode = result["metadata"]["mode"]
        dataset = result["metadata"]["dataset"]
        if mode == "lora":
            if result["metadata"].get("r", 8) != lora_rank:
                continue
        data[(mode, dataset)] = result
    return data


def get_datasets(data):
    return sorted({dataset for _, dataset in data.keys()})


def get_modes(data):
    return [m for m in MODE_ORDER if any(mode == m for mode, _ in data.keys())]


def bar_chart(results, metric_fn, title, ylabel, out_path, fmt=None):
    datasets = get_datasets(results)
    modes = get_modes(results)

    # Drop modes where every dataset yields None (e.g. baseline has no train time)
    active_modes = []
    for mode in modes:
        vals = [metric_fn(results.get((mode, d))) for d in datasets]
        if any(v is not None for v in vals):
            active_modes.append(mode)
    modes = active_modes

    x = np.arange(len(datasets))
    width = 0.8 / len(modes)

    fig, ax = plt.subplots(figsize=(max(5, len(datasets) * 2.5), 5))

    for i, mode in enumerate(modes):
        values = [metric_fn(results.get((mode, d))) for d in datasets]

        offset = (i - (len(modes) - 1) / 2) * width
        bars = ax.bar(
            [x[j] + offset for j, v in enumerate(values) if v is not None],
            [v for v in values if v is not None],
            width * 0.9,
            label=MODE_LABELS.get(mode, mode),
            color=MODE_COLORS.get(mode, None),
        )

        for bar, val in zip(bars, [v for v in values if v is not None]):
            label = fmt(val) if fmt else f"{val:.2f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    label, ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([d.upper() for d in datasets])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def parse_loss_history(log_history):
    train_losses, eval_losses = {}, {}
    for entry in log_history:
        epoch = entry.get("epoch")
        if epoch is None:
            continue
        if "loss" in entry and "eval_loss" not in entry and "train_runtime" not in entry:
            train_losses[epoch] = entry["loss"]
        if "eval_loss" in entry and "train_runtime" not in entry:
            eval_losses[epoch] = entry["eval_loss"]
    epochs = sorted(set(train_losses) | set(eval_losses))
    return (
        [e for e in epochs if e in train_losses],
        [train_losses[e] for e in epochs if e in train_losses],
        [e for e in epochs if e in eval_losses],
        [eval_losses[e] for e in epochs if e in eval_losses],
    )


def loss_plots(results, out_dir):
    datasets = get_datasets(results)
    trained_modes = [m for m in get_modes(results) if m != "baseline"]

    if not trained_modes:
        return

    for dataset in datasets:
        ncols = len(trained_modes)
        fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4), sharey=False)
        if ncols == 1:
            axes = [axes]

        fig.suptitle(f"Train vs Val Loss — {dataset.upper()}", fontsize=13)

        for ax, mode in zip(axes, trained_modes):
            entry = results.get((mode, dataset))
            if entry is None or not entry["training"].get("executed"):
                ax.set_visible(False)
                continue

            log_history = entry["training"]["log_history"]
            train_ep, train_loss, eval_ep, eval_loss = parse_loss_history(log_history)

            ax.plot(train_ep, train_loss, marker="o", label="Train loss",
                    color="#4878d0")
            ax.plot(eval_ep, eval_loss, marker="s", linestyle="--", label="Val loss",
                    color="#ee854a")
            ax.set_title(MODE_LABELS.get(mode, mode))
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
            ax.set_xticks(sorted(set(train_ep + eval_ep)))

        fig.tight_layout()
        path = os.path.join(out_dir, f"loss_{dataset}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate diagrams from result JSON files.")
    parser.add_argument("results_dir", help="Directory containing result JSON files")
    parser.add_argument("--out_dir", default=None,
                        help="Output directory for diagrams (default: results_dir)")
    parser.add_argument("--lora_rank", type=int, default=8,
                        help="Which LoRA rank file to use for the standard comparison charts (default: 8)")
    args = parser.parse_args()

    out_dir = args.out_dir or args.results_dir
    os.makedirs(out_dir, exist_ok=True)

    results = load_results(args.results_dir, lora_rank=args.lora_rank)
    if not results:
        print(f"No JSON files found in {args.results_dir}")
        return

    bar_chart(
        results,
        metric_fn=lambda e: e["training"]["peak_memory_mb"] if e else None,
        title="Peak VRAM per Dataset",
        ylabel="Peak VRAM (MB)",
        out_path=os.path.join(out_dir, "peak_vram.png"),
        fmt=lambda v: f"{v:.0f}",
    )

    bar_chart(
        results,
        metric_fn=lambda e: (
            e["training"]["time_sec"] / 60
            if e and e["training"].get("executed")
            else None
        ),
        title="Training Time per Dataset",
        ylabel="Time (min)",
        out_path=os.path.join(out_dir, "train_time.png"),
        fmt=lambda v: f"{v:.1f}m",
    )

    bar_chart(
        results,
        metric_fn=lambda e: e["eval"]["results"]["eval_accuracy"] * 100 if e else None,
        title="Final Validation Accuracy per Dataset",
        ylabel="Accuracy (%)",
        out_path=os.path.join(out_dir, "val_accuracy.png"),
        fmt=lambda v: f"{v:.1f}%",
    )

    loss_plots(results, out_dir)


if __name__ == "__main__":
    main()
