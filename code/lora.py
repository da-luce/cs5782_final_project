"""Custom LoRA implementation.

Re-implementation of the core ideas from Hu et al. 2021,
"LoRA: Low-Rank Adaptation of Large Language Models" (arXiv:2106.09685).

LoRA reparameterises a frozen linear layer W0 by adding a trainable low-rank
update:

    h = W0 x + b + (alpha / r) * B A x

where
    A in R^{r x in_features}    (random Gaussian init)
    B in R^{out_features x r}   (zeros init)
    r << min(in_features, out_features)

Because B is initialised to zero, the adapter contributes nothing at the start
of training, so the model's initial behaviour is identical to the pre-trained
base model (paper, Section 4.1).
"""

from typing import Iterable, List

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """A linear layer with a frozen base weight and a trainable low-rank update."""

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        alpha: float = 8.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank must be positive, got {r}")

        self.base_layer = base_layer
        self.r = r
        self.alpha = float(alpha)
        self.scaling = self.alpha / r

        for p in self.base_layer.parameters():
            p.requires_grad = False

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.normal_(self.lora_A, mean=0.0, std=1.0 / r)

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_out = self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return base_out + self.scaling * lora_out

    def extra_repr(self) -> str:
        return f"r={self.r}, alpha={self.alpha:g}, scaling={self.scaling:g}"


def freeze_module(module: nn.Module) -> None:
    """Set ``requires_grad = False`` on every parameter of ``module``."""
    for p in module.parameters():
        p.requires_grad = False


def inject_lora(
    root: nn.Module,
    target_module_names: Iterable[str] = ("query", "value"),
    r: int = 8,
    alpha: float = 8.0,
    dropout: float = 0.0,
) -> List[str]:
    """Replace every ``nn.Linear`` whose attribute name matches one in
    ``target_module_names`` with a :class:`LoRALinear` wrapping the original
    layer.

    For RoBERTa, passing ``("query", "value")`` adapts W_q and W_v in every
    self-attention block, matching the paper's main configuration (Section 4.2,
    Table 5).

    Returns the dotted paths of every replaced module (relative to ``root``).
    """
    targets = set(target_module_names)
    replaced: List[str] = []

    for module_name, module in list(root.named_modules()):
        for child_name, child in list(module.named_children()):
            if child_name in targets and isinstance(child, nn.Linear):
                lora_layer = LoRALinear(child, r=r, alpha=alpha, dropout=dropout)
                setattr(module, child_name, lora_layer)
                full = f"{module_name}.{child_name}" if module_name else child_name
                replaced.append(full)

    return replaced


def count_parameters(model: nn.Module) -> dict:
    """Return a small parameter-count report for the model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_trainable = sum(
        p.numel()
        for n, p in model.named_parameters()
        if p.requires_grad and ("lora_A" in n or "lora_B" in n)
    )
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "lora_trainable": lora_trainable,
        "non_lora_trainable": trainable - lora_trainable,
        "trainable_pct": (100.0 * trainable / total) if total else 0.0,
    }


def format_parameter_report(report: dict) -> str:
    return (
        f"Trainable: {report['trainable']:,} / {report['total']:,} "
        f"({report['trainable_pct']:.4f}%) | "
        f"LoRA: {report['lora_trainable']:,} | "
        f"Other trainable (e.g. classifier head): {report['non_lora_trainable']:,}"
    )
