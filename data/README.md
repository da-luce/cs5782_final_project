# Dataset Instructions

This project uses the **GLUE benchmark dataset**, specifically:

* SST-2 (Sentiment Analysis)
* MNLI (Natural Language Inference)

We do **not store the dataset in this repository** due to size and standard ML practice. Instead, it is automatically downloaded using the Hugging Face `datasets` library.

---

## How to Download the Data

The dataset will be automatically downloaded the first time you run the training scripts.

Example:

```python
from datasets import load_dataset

# SST-2
dataset = load_dataset("glue", "sst2")

# MNLI
dataset = load_dataset("glue", "mnli")
```

---

## Required Setup

Make sure dependencies are installed:

```bash
pip install datasets
```

or:

```bash
pip install -r requirements.txt
```

---

## Dataset Source

GLUE benchmark is publicly available through:

* Hugging Face Datasets
* Official GLUE benchmark website

---

## Notes

* No manual download is required.
* Data will be cached locally after first run (usually in `~/.cache/huggingface/datasets`).
