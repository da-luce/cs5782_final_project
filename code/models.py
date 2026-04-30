from transformers import AutoModelForSequenceClassification

from lora import apply_lora


def get_baseline_model(num_labels: int = 2):
    return AutoModelForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=num_labels,
    )


def get_lora_model(
    num_labels: int = 2,
    r: int = 8,
    lora_alpha: float = 128.0,
    lora_dropout: float = 0.1,
):
    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=num_labels,
    )

    return apply_lora(
        model,
        target_modules=["query", "value"],
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        modules_to_save=["classifier"],
    )
