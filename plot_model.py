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


def plot_layer_update_magnitude(
    model,
    proj_types=("query", "value"),
    title=r"Layer-wise LoRA Update Magnitude ($\|\Delta W\|_F$)",
):
    """
    Plots the Frobenius norm of ΔW = B·A·(α/r) for every transformer layer.
    """
    lora_layers = _get_lora_layers(model)

    layer_indices = sorted({
        int(n.split(".layer.")[1].split(".")[0])
        for n in lora_layers if ".layer." in n
    })

    norms = {pt: [] for pt in proj_types}
    for li in layer_indices:
        for pt in proj_types:
            key = f"roberta.encoder.layer.{li}.attention.self.{pt}"
            if key not in lora_layers:
                norms[pt].append(float("nan"))
                continue
            m = lora_layers[key]
            with torch.no_grad():
                delta_W = m.lora_B @ m.lora_A * m.scaling
                norms[pt].append(torch.norm(delta_W, p="fro").item())

    fig, ax = plt.subplots(figsize=(10, 5))
    for pt in proj_types:
        ax.plot(layer_indices, norms[pt], marker="o", label=pt)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Transformer Layer", fontsize=11)
    ax.set_ylabel(r"Frobenius Norm  $\|\Delta W\|_F$", fontsize=11)
    ax.set_xticks(layer_indices)
    ax.legend(title="Projection")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


# ── Helpers shared by magnitude and comparison plots ──────────────────────────

def _lora_layer_norms(model, proj_types):
    lora_layers = _get_lora_layers(model)
    layer_indices = sorted({
        int(n.split(".layer.")[1].split(".")[0])
        for n in lora_layers if ".layer." in n
    })
    norms = {pt: [] for pt in proj_types}
    for li in layer_indices:
        for pt in proj_types:
            key = f"roberta.encoder.layer.{li}.attention.self.{pt}"
            if key not in lora_layers:
                norms[pt].append(float("nan"))
                continue
            m = lora_layers[key]
            with torch.no_grad():
                delta_W = m.lora_B @ m.lora_A * m.scaling
                norms[pt].append(torch.norm(delta_W, p="fro").item())
    return layer_indices, norms


def _finetune_layer_norms(pretrained_model, finetuned_model, proj_types):
    def _attn_weights(m):
        return {
            n: p.detach()
            for n, p in m.named_parameters()
            if ".attention.self." in n
            and any(n.endswith(f"{pt}.weight") for pt in proj_types)
        }
    pre = _attn_weights(pretrained_model)
    ft  = _attn_weights(finetuned_model)
    layer_indices = sorted({
        int(n.split(".layer.")[1].split(".")[0])
        for n in pre if ".layer." in n
    })
    norms = {pt: [] for pt in proj_types}
    for li in layer_indices:
        for pt in proj_types:
            key = f"roberta.encoder.layer.{li}.attention.self.{pt}.weight"
            if key not in pre or key not in ft:
                norms[pt].append(float("nan"))
                continue
            norms[pt].append(torch.norm(ft[key] - pre[key], p="fro").item())
    return layer_indices, norms


def plot_finetune_layer_update_magnitude(
    pretrained_model,
    finetuned_model,
    proj_types=("query", "value"),
    title=r"Layer-wise Full Fine-Tune Update Magnitude ($\|\Delta W\|_F$)",
):
    """
    Plots the Frobenius norm of ΔW = W_finetuned − W_pretrained for each layer.
    """
    layer_indices, norms = _finetune_layer_norms(pretrained_model, finetuned_model, proj_types)

    fig, ax = plt.subplots(figsize=(10, 5))
    for pt in proj_types:
        ax.plot(layer_indices, norms[pt], marker="o", label=pt)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Transformer Layer", fontsize=11)
    ax.set_ylabel(r"Frobenius Norm  $\|\Delta W\|_F$", fontsize=11)
    ax.set_xticks(layer_indices)
    ax.legend(title="Projection")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_lora_vs_finetune_magnitude(
    lora_model,
    pretrained_model,
    finetuned_model,
    proj_types=("query", "value"),
    title=r"LoRA vs Full Fine-Tune: $\|\Delta W\|_F$ per Layer",
):
    """
    Overlays LoRA and full fine-tune update magnitudes on the same axes.
    Solid lines = full fine-tune; dashed lines = LoRA.
    """
    li_lora, norms_lora = _lora_layer_norms(lora_model, proj_types)
    li_ft,   norms_ft   = _finetune_layer_norms(pretrained_model, finetuned_model, proj_types)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, pt in enumerate(proj_types):
        c = colors[i % len(colors)]
        ax.plot(li_ft,   norms_ft[pt],   marker="o", color=c, linestyle="-",  label=f"{pt}  (full fine-tune)")
        ax.plot(li_lora, norms_lora[pt], marker="s", color=c, linestyle="--", label=f"{pt}  (LoRA)")

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Transformer Layer", fontsize=11)
    ax.set_ylabel(r"Frobenius Norm  $\|\Delta W\|_F$", fontsize=11)
    ax.set_xticks(li_ft)
    ax.legend(title="Projection · Method", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_lora_subspace_similarity(
    model,
    proj_types=("query", "value"),
    title=r"LoRA Subspace Alignment with Pretrained $W$",
):
    """
    For each layer, measures how much the LoRA update subspace (top-r left
    singular vectors of ΔW) aligns with the pretrained weight subspace (top-r
    left singular vectors of W_pre).

    Score = ||U_W[:, :r]ᵀ U_ΔW[:, :r]||_F² / r
      → 1.0: LoRA refines the model's most-used directions
      → 0.0: LoRA explores directions the pretrained model never used

    A horizontal reference line marks the expected score for a random rank-r
    update: r / d_out ≈ 0.01 for RoBERTa-base with r=8.
    """
    lora_layers = _get_lora_layers(model)

    layer_indices = sorted({
        int(n.split(".layer.")[1].split(".")[0])
        for n in lora_layers if ".layer." in n
    })

    similarities = {pt: [] for pt in proj_types}
    r_val, d_out_val = None, None

    for li in layer_indices:
        for pt in proj_types:
            key = f"roberta.encoder.layer.{li}.attention.self.{pt}"
            if key not in lora_layers:
                similarities[pt].append(float("nan"))
                continue
            m = lora_layers[key]
            r_val    = m.r
            d_out_val = m.original.weight.shape[0]

            with torch.no_grad():
                W_pre   = m.original.weight.float()
                delta_W = (m.lora_B @ m.lora_A * m.scaling).float()

                U_W,  _, _ = torch.linalg.svd(W_pre,   full_matrices=False)
                U_dW, _, _ = torch.linalg.svd(delta_W, full_matrices=False)

                U_W  = U_W[:,  :r_val]   # top-r directions of W_pre
                U_dW = U_dW[:, :r_val]   # top-r directions of ΔW

                sim = (U_W.T @ U_dW).pow(2).sum().item() / r_val

            similarities[pt].append(sim)

    fig, ax = plt.subplots(figsize=(10, 5))
    for pt in proj_types:
        ax.plot(layer_indices, similarities[pt], marker="o", label=pt)

    if r_val is not None and d_out_val is not None:
        baseline = r_val / d_out_val
        ax.axhline(
            baseline, color="gray", linestyle=":", alpha=0.8,
            label=f"random baseline  (r/d = {baseline:.3f})",
        )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Transformer Layer", fontsize=11)
    ax.set_ylabel("Subspace Similarity", fontsize=11)
    ax.set_xticks(layer_indices)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
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
