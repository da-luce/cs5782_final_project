from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model

def get_baseline_model():
    return AutoModelForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=2
    )

def get_lora_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=2
    )

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.1
    )

    return get_peft_model(model, config)