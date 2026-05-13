#!/usr/bin/env python3
"""Generate the parameter-efficiency Pareto plot for the poster.

For each dataset we plot validation accuracy against the number of
trainable parameters on a log-x axis. The pretrained zero-shot accuracy
is drawn as a horizontal reference line. The figure recreates Figure 2
of Hu et al. 2021 (LoRA) using our own training results.

Run:
    python make_pareto_plot.py results/2026.04.29-15.03.46-0c775f9-H100
"""

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np


MODE_LABELS = {
    "baseline": "Zero-shot",
    "lora": "LoRA",
    "finetune": "Full Fine-tune",
}
MODE_COLORS = {
    "baseline": "#9ca3af",
    "lora": "#ee854a",
    "finetune": "#4878d0",
}
MODE_MARKERS = {
    "lora": "o",
    "finetune": "^",
}
DATASET_TITLES = {"sst2": "SST-2", "mnli": "MNLI"}


def load_results(results_dir):
    out = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as f:
            r = json.load(f)
        key = (r["metadata"]["mode"], r["metadata"]["dataset"])
        out[key] = r
    return out


def fmt_params(n):
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.0f}K"
    return f"{n}"


def plot_panel(ax, results, dataset, full_ft_total=None):
    title = DATASET_TITLES.get(dataset, dataset.upper())

    zero_shot = results.get(("baseline", dataset))
    lora = results.get(("lora", dataset))
    ft = results.get(("finetune", dataset))

    accs = []

    if zero_shot is not None:
        zs_acc = zero_shot["eval"]["results"]["eval_accuracy"] * 100
        accs.append(zs_acc)
        ax.axhline(
            zs_acc,
            color=MODE_COLORS["baseline"],
            linestyle="--",
            linewidth=1.6,
            alpha=0.8,
            zorder=2,
        )
        ax.text(
            1.4e3,
            zs_acc + 0.6,
            f"Zero-shot ({zs_acc:.1f}%)",
            color="#4b5563",
            fontsize=10,
            fontstyle="italic",
        )

    points = []
    for mode in ("lora", "finetune"):
        r = results.get((mode, dataset))
        if r is None:
            continue
        acc = r["eval"]["results"]["eval_accuracy"] * 100
        n_train = r["model"]["trainable"]
        accs.append(acc)
        points.append((mode, n_train, acc))
        ax.scatter(
            n_train,
            acc,
            s=260,
            color=MODE_COLORS[mode],
            marker=MODE_MARKERS[mode],
            edgecolors="black",
            linewidth=1.6,
            zorder=4,
            label=MODE_LABELS[mode],
        )

    if len(points) == 2:
        xs = [p[1] for p in points]
        ys = [p[2] for p in points]
        ax.plot(xs, ys, color="#cbd5e1", linewidth=1.5, zorder=3, alpha=0.9)

    for mode, n_train, acc in points:
        if full_ft_total:
            pct = 100.0 * n_train / full_ft_total
            label = f"{MODE_LABELS[mode]}\n{fmt_params(n_train)} ({pct:.2f}%)"
        else:
            label = f"{MODE_LABELS[mode]}\n{fmt_params(n_train)}"
        # Both labels go below their points (the empty area between the
        # data points and the zero-shot line). LoRA hugs its point on the
        # left side; Full FT hugs its point on the right so neither label
        # clips the panel edges.
        if mode == "lora":
            offset = (-12, -22)
            ha = "right"
        else:
            offset = (12, -22)
            ha = "left"
        ax.annotate(
            label,
            xy=(n_train, acc),
            xytext=offset,
            textcoords="offset points",
            fontsize=10.5,
            ha=ha,
            va="top",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="white",
                edgecolor=MODE_COLORS[mode],
                linewidth=1.4,
                alpha=0.97,
            ),
            arrowprops=dict(
                arrowstyle="-",
                color=MODE_COLORS[mode],
                linewidth=1.1,
                alpha=0.7,
                shrinkA=0,
                shrinkB=8,
            ),
        )

    ax.set_xscale("log")
    ax.set_xlim(1e3, 5e8)
    if accs:
        lo = min(accs) - 4
        hi = max(accs) + 6
        ax.set_ylim(max(0, lo), min(100, hi))
    ax.set_xlabel("Trainable parameters (log scale)", fontsize=12)
    ax.set_ylabel("Validation accuracy (%)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, which="major", alpha=0.35, zorder=1)
    ax.grid(True, which="minor", alpha=0.15, zorder=1)


def make_plot(results, out_path, datasets=("sst2", "mnli")):
    full_ft_total = None
    for d in datasets:
        ft = results.get(("finetune", d))
        if ft is not None:
            full_ft_total = ft["model"]["total"]
            break

    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 5.2))
    if n == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        plot_panel(ax, results, dataset, full_ft_total=full_ft_total)

    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=len(handles),
            frameon=False,
            fontsize=11,
            bbox_to_anchor=(0.5, -0.02),
        )

    fig.suptitle(
        "LoRA matches full fine-tuning at <1% of trainable parameters",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results_dir",
        help="Directory containing baseline_*.json, lora_*.json, finetune_*.json",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output PNG path (default: <results_dir>/param_efficiency.png)",
    )
    args = parser.parse_args()

    results = load_results(args.results_dir)
    if not results:
        raise SystemExit(f"No JSON files in {args.results_dir}")

    out = args.out or os.path.join(args.results_dir, "param_efficiency.png")
    make_plot(results, out)


if __name__ == "__main__":
    main()
