# LoRA Re-implementation (CS 4/5782 Final Project)

## 1. Introduction

This repository contains our re-implementation of *LoRA: Low-Rank Adaptation of Large Language Models* (Hu et al., 2021). Rather than fine-tuning all parameters of a large pre-trained model, LoRA freezes the original weights and injects trainable low-rank matrices into the attention layers — reducing trainable parameters by orders of magnitude while maintaining comparable performance.

We study whether a from-scratch LoRA implementation can match full fine-tuning on RoBERTa-base, targeting the results reported in Table 2 of the original paper.

## 2. Chosen Result

We reproduce the RoBERTa-base results from **Table 2** of the paper, comparing full fine-tuning vs. LoRA (r=8) on two GLUE tasks:

| Task | Metric | Full FT (paper) | LoRA (paper) |
|---|---|---|---|
| SST-2 | Accuracy | 94.8 | 94.9 |
| MNLI | Accuracy | 87.6 | 87.5 |

LoRA reduces trainable parameters from ~125M (100%) to ~0.3M (~0.3%) with no meaningful accuracy loss — the core efficiency claim of the paper.

## 3. Repository Structure

```
code/           Custom LoRA implementation and training script
  lora.py       LoRALinear layer, apply_lora(), count_parameters()
  models.py     get_baseline_model() and get_lora_model() using lora.py
  train.py      Training script (--mode baseline|lora, --dataset sst2|mnli)
data/           Dataset info — data is auto-downloaded via HuggingFace
results/        JSON result files produced per experiment run
poster/         Poster PDF (added before presentation)
report/         2-page summary report PDF (added before May 12)
demo.ipynb      Colab notebook — runs all experiments and prints comparison table
requirements.txt
```

## 4. Re-implementation Details

- **Model:** `roberta-base` (125M parameters) via HuggingFace Transformers
- **Custom LoRA:** implemented from scratch in `code/lora.py` — no PEFT library used
  - Low-rank matrices A (rank × d, kaiming-uniform init) and B (d × rank, zero init) injected into the `query` and `value` projection layers of all 12 transformer blocks
  - Rank r=8, scaling α=16, dropout=0.1 — matching the paper's RoBERTa-base settings
  - All base model weights frozen; only A, B, and the classification head are trained
- **Datasets:** SST-2 (2-class sentiment) and MNLI (3-class NLI) from GLUE via HuggingFace `datasets`
- **Subset:** 2,000 training samples and 500 validation samples per experiment for Colab feasibility
- **Evaluation metric:** accuracy on the validation set (SST-2: `validation`, MNLI: `validation_matched`)
- **Training:** 2 epochs, batch size 8; lr=2e-5 for baseline, lr=2e-4 for LoRA

## 5. How to Run

### Option A — Google Colab (recommended)

1. Open [`demo.ipynb`](./demo.ipynb) in Google Colab
2. Set runtime to **GPU** (Runtime → Change runtime type → T4)
3. Run all cells — experiments run sequentially and print a comparison table at the end

### Option B — Local

```shell
git clone https://github.com/da-luce/cs5782_final_project.git
cd cs5782_final_project
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run individual experiments from the code/ directory
cd code
python train.py --mode baseline --dataset sst2
python train.py --mode lora     --dataset sst2
python train.py --mode baseline --dataset mnli
python train.py --mode lora     --dataset mnli
```

**Computational requirements:** A GPU with ≥8 GB VRAM (e.g., Colab T4 at 16 GB). Each experiment takes approximately 5–15 minutes on a T4.

Results are saved as JSON files in `results/`.

## 6. Results / Insights

*To be filled in after experiments are run. See `results/` for raw JSON files.*

## 7. Conclusion

*To be filled in after experiments are run.*

## 8. References

- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685.
- Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., & Stoyanov, V. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach.* arXiv:1907.11692.
- Wang, A., Singh, A., Michael, J., Hill, F., Levy, O., & Bowman, S. (2018). *GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding.* arXiv:1804.07461.
- Wolf, T., et al. (2020). *HuggingFace's Transformers: State-of-the-art Natural Language Processing.* arXiv:1910.03771.

## 9. Acknowledgements

This project was completed as part of CS 4/5782 (Deep Learning) at Cornell University, Spring 2026. We thank the course staff for their guidance.
