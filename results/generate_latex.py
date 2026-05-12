import json
import os
import glob
import argparse

# Paper reference results from Table 2 of the LoRA paper (RoBERTa-base)
PAPER_RESULTS = [
    {"source": "paper", "mode": "finetune", "dataset": "sst2",  "method": "Full fine-tuning (paper)", "acc": 94.8, "trainable_m": None, "pct": None, "time_m": None, "ckpt": "---", "vram": "---"},
    {"source": "paper", "mode": "lora",     "dataset": "sst2",  "method": "LoRA $r=8$ (paper)",       "acc": 94.9, "trainable_m": None, "pct": None, "time_m": None, "ckpt": "---", "vram": "---"},
    {"source": "paper", "mode": "finetune", "dataset": "mnli",  "method": "Full fine-tuning (paper)", "acc": 87.6, "trainable_m": None, "pct": None, "time_m": None, "ckpt": "---", "vram": "---"},
    {"source": "paper", "mode": "lora",     "dataset": "mnli",  "method": "LoRA $r=8$ (paper)",       "acc": 87.5, "trainable_m": None, "pct": None, "time_m": None, "ckpt": "---", "vram": "---"},
]

def fmt(val, fmt_str, suffix=""):
    if val is None:
        return "---"
    return format(val, fmt_str) + suffix

def generate_latex_table(folder):
    json_files = glob.glob(os.path.join(folder, "*.json"))
    if not json_files:
        print(f"No JSON files found in {folder}")
        return

    rows = []
    for path in json_files:
        with open(path) as f:
            data = json.load(f)

        meta = data.get("metadata", {})
        mode = meta.get("mode", data.get("mode", "unknown"))
        dataset = meta.get("dataset", data.get("dataset", "unknown"))

        if mode == "baseline":
            method_str = "Baseline (untrained)"
        elif mode == "finetune":
            method_str = "Full fine-tuning (ours)"
        elif mode == "lora":
            method_str = "LoRA $r=8$ (ours)"
        else:
            method_str = mode

        model_info = data.get("model", data.get("trainable_params", {}))
        trainable = model_info.get("trainable", 0)
        trainable_m = trainable / 1e6
        trainable_pct = model_info.get("trainable_pct", 0)

        training = data.get("training", {})
        time_sec = training.get("time_sec", data.get("training_time_sec", 0))
        time_m = time_sec / 60 if time_sec else None

        ckpt_mb = training.get("checkpoint_size_mb", data.get("checkpoint_size_mb"))
        vram_mb = training.get("peak_memory_mb", data.get("peak_memory_mb"))

        eval_results = data.get("eval", {}).get("results", data.get("eval_results", {}))
        acc = eval_results.get("eval_accuracy", 0) * 100

        rows.append({
            "source": "ours",
            "mode": mode,
            "dataset": dataset,
            "method": method_str,
            "acc": acc,
            "trainable_m": trainable_m,
            "pct": trainable_pct,
            "time_m": time_m if time_m else None,
            "ckpt": f"{ckpt_mb:.1f}M" if isinstance(ckpt_mb, (int, float)) else "---",
            "vram": f"{vram_mb:.1f}M" if isinstance(vram_mb, (int, float)) else "---",
        })

    all_rows = PAPER_RESULTS + rows

    # Sort: dataset, then paper-before-ours, then mode order
    mode_order = {"baseline": 0, "finetune": 1, "lora": 2}
    source_order = {"paper": 0, "ours": 1}
    all_rows.sort(key=lambda x: (x["dataset"], mode_order.get(x["mode"], 9), source_order.get(x["source"], 9)))

    latex = []
    latex.append("\\documentclass[11pt]{article}")
    latex.append("\\usepackage[margin=1in]{geometry}")
    latex.append("\\usepackage{booktabs}")
    latex.append("\\begin{document}")
    latex.append("")
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\small")
    latex.append("\\begin{tabular}{llrrrrrr}")
    latex.append("  \\toprule")
    latex.append("  Method & Dataset & Accuracy & Params & \\% & Time & Ckpt & Peak VRAM \\\\")
    latex.append("  \\midrule")

    prev_dataset = None
    for r in all_rows:
        dataset_upper = r["dataset"].upper()
        if prev_dataset is not None and r["dataset"] != prev_dataset:
            latex.append("  \\midrule")
        prev_dataset = r["dataset"]

        acc_str = fmt(r["acc"], ".1f", "\\%")
        trainable_str = fmt(r["trainable_m"], ".1f", "M")
        pct_str = fmt(r["pct"], ".2f", "\\%")
        time_str = fmt(r["time_m"], ".1f", "m")

        latex.append(
            f"  {r['method']:<30} & {dataset_upper:<6} & {acc_str:>7} & "
            f"{trainable_str:>7} & {pct_str:>7} & {time_str:>7} & "
            f"{r['ckpt']:>8} & {r['vram']:>8} \\\\"
        )

    latex.append("  \\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\caption{RoBERTa-base results on GLUE tasks. Paper results from Table~2 of \\citet{hu2022lora}.}")
    latex.append("\\label{tab:results}")
    latex.append("\\end{table}")
    latex.append("")
    latex.append("\\end{document}")

    print("\n" + "\n".join(latex) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LaTeX table from a specific results folder.")
    parser.add_argument("folder", type=str, help="Path to a run folder containing JSON result files.")
    args = parser.parse_args()

    generate_latex_table(args.folder)
