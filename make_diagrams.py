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


def load_lora_ranks(results_dir):
    """Group every lora_*_r*.json by dataset, sorted by rank."""
    by_dataset = {}
    for path in glob.glob(os.path.join(results_dir, "lora_*_r*.json")):
        with open(path) as f:
            result = json.load(f)
        dataset = result["metadata"]["dataset"]
        rank = result["metadata"]["r"]
        by_dataset.setdefault(dataset, []).append((rank, result))
    for ds in by_dataset:
        by_dataset[ds].sort(key=lambda x: x[0])
    return by_dataset


def fmt_params(n):
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    if n >= 1_000:
        return f"{n / 1e3:.0f}K"
    return f"{n}"


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


def rank_ablation_plot(results_dir, out_dir, baselines=None):
    """Plot LoRA validation accuracy vs. rank, one panel per dataset.

    Reads every lora_<dataset>_r<rank>.json in results_dir; if baselines is
    given (the standard load_results dict), draws zero-shot and full-FT
    horizontal reference lines per panel.
    """
    by_dataset = load_lora_ranks(results_dir)
    if not by_dataset:
        return

    datasets = sorted(by_dataset.keys())
    ncols = len(datasets)
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5), sharey=False)
    if ncols == 1:
        axes = [axes]

    fig.suptitle("LoRA accuracy vs. rank", fontsize=13)

    for ax, dataset in zip(axes, datasets):
        ranks = [r for r, _ in by_dataset[dataset]]
        accs = [d["eval"]["results"]["eval_accuracy"] * 100 for _, d in by_dataset[dataset]]
        params = [d["model"]["trainable"] for _, d in by_dataset[dataset]]

        refs = []
        if baselines:
            for ref_mode, ref_label, style in [
                ("baseline", "Zero-shot", ":"),
                ("finetune", "Full FT", "--"),
            ]:
                entry = baselines.get((ref_mode, dataset))
                if entry is None:
                    continue
                val = entry["eval"]["results"]["eval_accuracy"] * 100
                color = MODE_COLORS[ref_mode]
                ax.axhline(val, color=color, linestyle=style, linewidth=1.6,
                           alpha=0.8, zorder=2)
                refs.append((ref_label, val, color))

        ax.plot(ranks, accs, color=MODE_COLORS["lora"], linewidth=2.5, zorder=3)
        ax.scatter(ranks, accs, s=160, color=MODE_COLORS["lora"],
                   edgecolors="black", linewidth=1.4, zorder=4,
                   label=MODE_LABELS["lora"])

        for r, acc, n in zip(ranks, accs, params):
            ax.annotate(f"{acc:.1f}%\n{fmt_params(n)}",
                        xy=(r, acc), xytext=(0, -28),
                        textcoords="offset points",
                        fontsize=9, ha="center", va="top", color="#374151")

        ax.set_xscale("log", base=2)
        ax.set_xticks(ranks)
        ax.set_xticklabels([str(r) for r in ranks])
        ax.set_xlabel("LoRA rank r (log₂)")
        ax.set_ylabel("Validation accuracy (%)")
        ax.set_title(dataset.upper())
        ax.grid(True, which="major", alpha=0.35, zorder=1)

        all_y = list(accs) + [v for _, v, _ in refs]
        span = max(all_y) - min(all_y)
        pad_top = max(3.0, 0.05 * span + 1.0)
        pad_bot = max(6.0, 0.18 * span + 3.0)
        ax.set_ylim(max(0, min(all_y) - pad_bot), min(100, max(accs) + pad_top))

        ref_y = 0.04
        for ref_label, val, color in refs:
            ax.text(0.98, ref_y, f"{ref_label}: {val:.1f}%",
                    transform=ax.transAxes, ha="right", va="bottom",
                    color=color, fontsize=9, fontweight="bold", fontstyle="italic")
            ref_y += 0.05

    fig.tight_layout()
    path = os.path.join(out_dir, "rank_ablation.png")
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

    rank_ablation_plot(args.results_dir, out_dir, baselines=results)


if __name__ == "__main__":
    main()
