# Results

This directory contains the JSON result logs produced by `code/train.py` for each experiment. 

Each file is named using the `{mode}_{dataset}.json` convention (e.g., `baseline_sst2.json`, `lora_mnli.json`) and records the hyperparameter configurations, parameter efficiency stats, independent execution times, and final evaluation metrics for that specific run.

Result files are committed here after experiments are executed in the Colab environment to preserve historical data without needing to re-run the training pipeline.

### JSON Schema Structure

All result files follow this nested format:

```json
{
    "metadata": {
        "mode": "lora",
        "dataset": "sst2",
        "device": "cuda"
    },
    "model": {
        "total": 125000000,
        "trainable": 300000,
        "trainable_pct": 0.24,
        "frozen": 124700000
    },
    "training": {
        "executed": true,
        "samples": 5000,
        "epochs": 5,
        "learning_rate": 0.0002,
        "batch_size": 8,
        "time_sec": 345.67,
        "model_output_dir": "./models/lora_sst2"
    },
    "eval": {
        "samples": 500,
        "time_sec": 12.34,
        "results": {
            "eval_loss": 0.315,
            "eval_accuracy": 0.884,
            "eval_runtime": 12.33,
            "eval_samples_per_second": 40.5,
            "eval_steps_per_second": 5.1
        }
    }
}

Notice that for the baseline, we do not train the model so the training key should be ignored or treated as null metadata.