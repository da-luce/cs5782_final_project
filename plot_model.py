import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def _get_lora_layers(model):
    return {n: m for n, m in model.named_modules() if type(m).__name__ == "LoRALinear"}


def plot_lora_heatmaps(
    model,
    layer_indices=(0, 5, 11),
    proj_types=("query", "value"),
    cmap=None,
    title=r"LoRA Weight Updates ($\Delta W$) Heatmaps",
):
    """
    Plots ΔW = B·A·(α/r) for each specified layer and projection type.
    """
    lora_layers = _get_lora_layers(model)

    if cmap is None:
        cmap = "RdBu_r"

    fig, axes = plt.subplots(
        len(proj_types), len(layer_indices),
        figsize=(18, 8),
        squeeze=False,
    )
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for row, pt in enumerate(proj_types):
        for col, li in enumerate(layer_indices):
            key = f"roberta.encoder.layer.{li}.attention.self.{pt}"

            if key not in lora_layers:
                print(f"Warning: {key} not found in model.")
                continue

            m = lora_layers[key]

            with torch.no_grad():
                delta_W = (m.lora_B @ m.lora_A * m.scaling).cpu().numpy()

            limit = np.percentile(np.abs(delta_W), 99)

            ax = axes[row, col]
            im = ax.imshow(delta_W, cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
            ax.set_title(f"Layer {li}  ·  {pt}", fontsize=11)
            ax.set_xlabel("Input dim", fontsize=8)
            if col == 0:
                ax.set_ylabel("Output dim", fontsize=8)
            ax.tick_params(labelsize=7)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


def plot_lora_bottleneck(
    model,
    layer_index=5,
    proj_type="query",
    cmap=None,
    title=None,
):
    """
    Plots the raw A and B matrices side-by-side for a single LoRA layer,
    making the low-rank bottleneck structure visually obvious.
    """
    lora_layers = _get_lora_layers(model)

    if cmap is None:
        cmap = "RdBu_r"

    key = f"roberta.encoder.layer.{layer_index}.attention.self.{proj_type}"
    if key not in lora_layers:
        print(f"Warning: {key} not found in model.")
        return

    m = lora_layers[key]

    with torch.no_grad():
        A = m.lora_A.cpu().numpy()  # (r, d_in)
        B = m.lora_B.cpu().numpy()  # (d_out, r)

    r, d_in  = A.shape
    d_out, _ = B.shape

    full_params = d_in * d_out
    lora_params = r * d_in + d_out * r

    if title is None:
        title = (
            f"Low-Rank Bottleneck  |  Layer {layer_index} · {proj_type}  |  r = {r}\n"
            f"A + B: {lora_params:,} params  vs  full W: {full_params:,} params  "
            f"({100 * (1 - lora_params / full_params):.0f}% fewer)"
        )

    fig, (ax_A, ax_B) = plt.subplots(
        1, 2,
        figsize=(16, 6),
        gridspec_kw={"width_ratios": [5, 1]},
    )
    fig.suptitle(title, fontsize=13, fontweight="bold")

    vA = np.percentile(np.abs(A), 99)
    im_A = ax_A.imshow(A, cmap=cmap, vmin=-vA, vmax=vA, aspect="auto")
    ax_A.set_title(f"Matrix A  ({r} × {d_in})\nCompresses input → rank", fontsize=12)
    ax_A.set_xlabel(f"Input features  (d_in = {d_in})", fontsize=10)
    ax_A.set_ylabel(f"Rank  (r = {r})", fontsize=10)
    ax_A.set_yticks(range(r))
    plt.colorbar(im_A, ax=ax_A, fraction=0.015, pad=0.02)

    vB = np.percentile(np.abs(B), 99)
    im_B = ax_B.imshow(B, cmap=cmap, vmin=-vB, vmax=vB, aspect="auto")
    ax_B.set_title(f"Matrix B  ({d_out} × {r})\nExpands rank → output", fontsize=12)
    ax_B.set_xlabel(f"Rank  (r = {r})", fontsize=10)
    ax_B.set_ylabel(f"Output features  (d_out = {d_out})", fontsize=10)
    ax_B.set_xticks(range(r))
    plt.colorbar(im_B, ax=ax_B, fraction=0.15, pad=0.04)

    plt.tight_layout()
    plt.show()
