import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer
import numpy as np
import evaluate

from models import get_baseline_model, get_lora_model

TRAIN_SAMPLES = 2000
VAL_SAMPLES = 500

def run_experiment(mode, dataset_name):

    # dataset
    dataset = load_dataset("glue", dataset_name)

    # select a subset for faster training
    dataset["train"] = dataset["train"].select(range(TRAIN_SAMPLES))
    dataset["validation"] = dataset["validation"].select(range(VAL_SAMPLES))

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    def tokenize(batch):
        return tokenizer(batch["sentence"], truncation=True, padding="max_length")

    # Important: tokenize the dataset after selecting the subset, 
    # otherwise it will take a long time to tokenize the entire dataset
    dataset = dataset.map(tokenize, batched=True)

    # model switch
    model = get_baseline_model() if mode == "baseline" else get_lora_model()

    # metrics
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=labels)

    # training args
    args = TrainingArguments(
        output_dir=f"results/{mode}",
        eval_strategy="epoch",
        learning_rate=2e-5 if mode == "baseline" else 2e-4,
        per_device_train_batch_size=8,
        num_train_epochs=2
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.evaluate()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, required=True, choices=["baseline", "lora"])
    parser.add_argument("--dataset", type=str, default="sst2")

    args = parser.parse_args()

    run_experiment(args.mode, args.dataset)


if __name__ == "__main__":
    main()