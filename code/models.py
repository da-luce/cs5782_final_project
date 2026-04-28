from typing import List, Tuple

from transformers import AutoModelForSequenceClassification

from lora import freeze_module, inject_lora


def get_baseline_model(num_labels: int = 2):
    """Plain RoBERTa-base for full fine-tuning."""
    return AutoModelForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=num_labels,
    )


def get_lora_model(
    num_labels: int = 2,
    r: int = 8,
    alpha: float = 8.0,
    dropout: float = 0.1,
    target_modules: Tuple[str, ...] = ("query", "value"),
) -> Tuple["AutoModelForSequenceClassification", List[str]]:
    """RoBERTa-base with the backbone frozen and LoRA injected into the
    self-attention query/value projections.

    The classification head (``model.classifier``) stays trainable — this is
    standard practice when adapting a pre-trained encoder to a new task and is
    consistent with the parameter counts reported in Table 2 of the paper
    (which only count the LoRA matrices).
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=num_labels,
    )

    freeze_module(model.roberta)

    replaced = inject_lora(
        model.roberta,
        target_module_names=target_modules,
        r=r,
        alpha=alpha,
        dropout=dropout,
    )

    return model, replaced
