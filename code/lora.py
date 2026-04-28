import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """nn.Linear wrapped with low-rank adaptation matrices A and B.

    During a forward pass the output is:
        W x + (B A x) * (alpha / r)
    where W is the frozen pretrained weight, A is initialized with
    kaiming-uniform (approximates the Gaussian init from the paper),
    and B is initialized to zero so the LoRA branch starts as a no-op.
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.1,
    ):
        super().__init__()
        assert r > 0, "LoRA rank r must be a positive integer"

        self.original = original_linear
        self.r = r
        self.scaling = lora_alpha / r

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # Trainable low-rank matrices
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        self.lora_dropout = nn.Dropout(p=lora_dropout)

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Freeze the original pretrained weights
        for param in self.original.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.original(x)
        lora_out = (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out


def apply_lora(
    model: nn.Module,
    target_modules: list[str],
    r: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.1,
    modules_to_save: list[str] | None = None,
) -> nn.Module:
    """Inject LoRA into every nn.Linear whose name ends with a target token.

    Steps:
      1. Freeze all base model parameters.
      2. Replace matching Linear layers with LoRALinear (which keeps its own
         lora_A / lora_B trainable).
      3. Re-enable gradients for any module listed in modules_to_save
         (e.g. the task-specific classification head).
    """
    if modules_to_save is None:
        modules_to_save = ["classifier"]

    # Step 1: freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Step 2: replace target Linear layers
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(name == t or name.endswith(f".{t}") for t in target_modules):
            continue

        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        attr = parts[-1]

        setattr(parent, attr, LoRALinear(module, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout))

    # Step 3: unfreeze modules_to_save
    for name, module in model.named_modules():
        if any(name == m or name.startswith(f"{m}.") for m in modules_to_save):
            for param in module.parameters():
                param.requires_grad = True

    return model


def count_parameters(model: nn.Module) -> dict:
    """Return total, trainable, frozen parameter counts and trainable %."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "trainable_pct": round(100.0 * trainable / total, 4) if total > 0 else 0.0,
    }
