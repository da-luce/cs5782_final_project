import argparse
import json
import time

from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer
import numpy as np
import evaluate
import torch

from models import get_baseline_model, get_lora_model

TRAIN_SAMPLES = 2000
VAL_SAMPLES = 500


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def run_experiment(mode, dataset_name):

    device = get_device()
    print(f"Using device: {device}")

    start_time = time.time()

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
        num_train_epochs=2,
        no_cuda=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics
    )

    trainer.train()

    eval_result = trainer.evaluate()

    trainer.save_model(f"./models/{mode}")

    end_time = time.time()

    info = {
        "mode": mode,
        "dataset": dataset_name,
        "device": device,
        "train_samples": TRAIN_SAMPLES,
        "val_samples": VAL_SAMPLES,
        "epochs": 2,
        "learning_rate": 2e-5 if mode == "baseline" else 2e-4,
        "batch_size": 8,
        "eval_results": eval_result,
        "training_time_sec": end_time - start_time,
        "model_output_dir": f"./models/{mode}"
    }

    os.makedirs("./models/info", exist_ok=True)

    info_path = f"./models/info/{mode}_{dataset_name}.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)

    print(f"\nSaved model to ./models/{mode}")
    print(f"Saved run info to {info_path}")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, required=True, choices=["baseline", "lora"])
    parser.add_argument("--dataset", type=str, default="sst2")

    args = parser.parse_args()

    run_experiment(args.mode, args.dataset)


if __name__ == "__main__":
    main()