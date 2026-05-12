# LoRA Re-implementation (CS 4/5782 Final Project)

## 1. Introduction

This repository contains a from-scratch re-implementation of **LoRA: Low-Rank Adaptation of Large Language Models** (Hu et al., 2021) as a CS 4/5782 (Deep Learning) final project at Cornell University.

LoRA freezes the pre-trained weights of a large model and injects trainable low-rank decomposition matrices into the attention layers, cutting trainable parameters by orders of magnitude with no meaningful accuracy loss. Formally, the weight update is represented as $\Delta W = BA$ where rank $r \ll d$, and the forward pass becomes $h = W_0 x + \frac{\alpha}{r}BAx$. This avoids the prohibitive memory and compute costs of full fine-tuning while preserving inference speed and the original context window.

## 2. Chosen Result

We reproduce the RoBERTa-base results from **Table 2** of the paper, comparing full fine-tuning vs. LoRA ($r=8$) on two GLUE benchmarks: SST-2 (binary sentiment classification) and MNLI (3-class natural language inference).

| Task  | Method           | Accuracy (paper) | Accuracy (ours) | Trainable Params |
| ----- | ---------------- | ---------------- | --------------- | ---------------- |
| SST-2 | Full fine-tuning | 94.8%            | 92.7%           | 124.6M (100%)    |
| SST-2 | LoRA $r=8$       | 94.9%            | **93.5%**       | 0.9M (0.71%)     |
| MNLI  | Full fine-tuning | 87.6%            | 84.4%           | 124.6M (100%)    |
| MNLI  | LoRA $r=8$       | 87.5%            | 84.2%           | 0.9M (0.71%)     |

This result validates the core LoRA claim: a low-rank update with ~0.3M adapter parameters (plus ~0.6M for the classification head) matches full fine-tuning accuracy while training fewer than 1% of total parameters.

## 3. Repository Structure

```
code/
  demo.ipynb            Colab notebook — runs all experiments and prints comparison table
  lora/
    lora.py             LoRALinear layer, apply_lora(), count_parameters()
    models.py           get_baseline_model() and get_lora_model()
    train.py            Training script (--mode baseline|finetune|lora, --dataset sst2|mnli, --rank r)
  misc/
    analyze_truncation.py  Sequence-length statistics justifying 128-token truncation
  plot/
    make_diagrams.py    Plot generation for results figures
    plot_model.py       LoRA weight heatmap visualization
data/                 Dataset notes — data is auto-downloaded via HuggingFace
results/              JSON result files produced per experiment run
poster/               Poster PDF
report/               Final report PDF (report.tex + compiled PDF)
requirements.txt
```

## 4. Re-implementation Details

**Model:** `roberta-base` (125M parameters, 12-layer transformer) via HuggingFace Transformers.

**Custom LoRA** (`code/lora.py`) — no PEFT library used:
- `LoRALinear` augments selected `nn.Linear` layers with matrices $A \in \mathbb{R}^{r \times d}$ (Kaiming-uniform init) and $B \in \mathbb{R}^{d \times r}$ (zero init), ensuring $\Delta W = 0$ at initialization.
- LoRA is injected into the `query` and `value` projections of all 12 attention blocks.
- Rank $r=8$, scaling $\alpha=16$, dropout $0.1$ — matching Table 2 of the paper.
- All base model weights are frozen; only the LoRA matrices and the classification head are trained.

**Datasets:** SST-2 and MNLI from GLUE via HuggingFace `datasets`.

**Evaluation metric:** Validation accuracy (`validation` split for SST-2, `validation_matched` for MNLI).

**Modifications relative to the paper:**
1. We train on 50,000 random training examples per task (vs. 67K/393K in the paper) due to compute constraints — this accounts for the ~2–3% accuracy gap.
2. Sequences are truncated to 128 tokens rather than RoBERTa's default of 512. Sequence-length analysis (`code/analyze_truncation.py`) shows 100% of SST-2 and 99.7% of MNLI examples fit within 128 tokens, so no meaningful information is lost.

## 5. Reproduction Steps

### Option A — Google Colab (recommended)

1. Open [`demo.ipynb`](./code/demo.ipynb) in Google Colab.
2. Set runtime to **GPU** (Runtime -> Change runtime type -> T4 or better).
3. Run all cells--experiments run sequentially and print a comparison table at the end.

### Option B — Local / Cluster

```shell
git clone https://github.com/da-luce/cs5782_final_project.git
cd cs5782_final_project
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run from code/lora/ directory
cd code/lora

# Baseline (no fine-tuning, evaluation only)
python train.py --mode baseline --dataset sst2
python train.py --mode baseline --dataset mnli

# Full fine-tuning
python train.py --mode finetune --dataset sst2
python train.py --mode finetune --dataset mnli

# LoRA (r=8 by default)
python train.py --mode lora --dataset sst2
python train.py --mode lora --dataset mnli

# Rank ablation
python train.py --mode lora --rank 2  --dataset sst2
python train.py --mode lora --rank 4  --dataset sst2
python train.py --mode lora --rank 8  --dataset sst2
python train.py --mode lora --rank 16 --dataset sst2
```

Optional flags for `train.py`: `--train_samples` (default 50000), `--val_samples` (default 500), `--epochs` (default 5).

Results are saved as JSON files in `results/`.

**Computational requirements:** A GPU with >=8 GB VRAM. Each main experiment takes ~7–9 minutes on an NVIDIA H100; a Colab T4 will be slower (~15–20 min per run). CPU-only is not recommended.

## 6. Results / Insights

Our results (trained on 50K examples, 3 epochs, single H100) closely track the paper's relative trends:

| Method               | Dataset | Accuracy  | Trainable Params | Peak VRAM   | Train Time  |
| -------------------- | ------- | --------- | ---------------- | ----------- | ----------- |
| Baseline (untrained) | MNLI    | 33.4%     | 124.6M           | 543 MB      | —           |
| Full fine-tuning     | MNLI    | 84.4%     | 124.6M           | 2135 MB     | 9.2 min     |
| LoRA $r=8$           | MNLI    | 84.2%     | 0.9M             | **1045 MB** | **7.0 min** |
| Baseline (untrained) | SST-2   | 50.9%     | 124.6M           | 543 MB      | —           |
| Full fine-tuning     | SST-2   | 92.7%     | 124.6M           | 2135 MB     | 8.9 min     |
| LoRA $r=8$           | SST-2   | **93.5%** | 0.9M             | **1045 MB** | **6.6 min** |

Key insights beyond the paper:

- **Rank ablation ($r \in \{2, 4, 8, 16\}$):** Accuracy is flat across all ranks (~78–80% on MNLI, ~92–93% on SST-2), suggesting the adaptation signal for these GLUE benchmarks is intrinsically low-dimensional — even $r=2$ suffices.
- **Implicit regularization:** On MNLI, full fine-tuning shows validation loss divergence by epoch 2, while LoRA's constrained update space keeps validation loss stable throughout training. LoRA's low-rank bottleneck acts as a structural prior, not just a parameter reduction trick.

## 7. Conclusion

Our re-implementation confirms LoRA's central claim: low-rank adaptation matches full fine-tuning with under 1% of trainable parameters, cutting peak VRAM usage by ~2× and training time by ~25%. The rank ablation extends the paper's findings, showing that $r=2$ already captures the full adaptation signal for SST-2 and MNLI. An unexpected result is LoRA's role as an implicit regularizer, preventing the overfitting observed in full fine-tuning on MNLI. Together, these results suggest that LoRA's low-rank constraint is a meaningful structural prior, not merely a computational shortcut.

## 8. References

- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685.
- Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., & Stoyanov, V. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach.* arXiv:1907.11692.
- Wang, A., Singh, A., Michael, J., Hill, F., Levy, O., & Bowman, S. (2018). *GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding.* arXiv:1804.07461.
- Wolf, T., et al. (2020). *HuggingFace's Transformers: State-of-the-art Natural Language Processing.* arXiv:1910.03771.

## 9. Acknowledgements

This project was completed as part of CS 4/5782 (Deep Learning) at Cornell University, Spring 2026, by Jason Dong, Sennet Senadheera, and Dalton Luce. We thank the course staff for their guidance and feedback throughout the semester.
