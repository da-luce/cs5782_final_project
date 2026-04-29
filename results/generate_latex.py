import json
import os
import glob
import argparse

def generate_latex_table(results_dir):
    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    if not json_files:
        print(f"No JSON files found in {results_dir}")
        return
        
    rows = []
    for path in json_files:
        with open(path) as f:
            data = json.load(f)
            
        mode = data.get("mode", "unknown")
        dataset = data.get("dataset", "unknown")
        
        # Format mode
        if mode == "baseline":
            method_str = "Full fine-tuning"
        elif mode == "lora":
            method_str = "LoRA ($r=8$)"
        else:
            method_str = mode
            
        # Extract metrics
        acc = data.get("eval_results", {}).get("eval_accuracy", 0) * 100
        
        params = data.get("trainable_params", {})
        trainable = params.get("trainable", 0)
        trainable_m = trainable / 1e6
        
        trainable_pct = params.get("trainable_pct", 0)
        
        time_sec = data.get("training_time_sec", 0)
        time_m = time_sec / 60
        
        # Extract new metrics if available
        ckpt_mb = data.get("checkpoint_size_mb", "---")
        if isinstance(ckpt_mb, (int, float)):
            ckpt_mb = f"{ckpt_mb:.1f}M"
            
        vram_mb = data.get("peak_memory_mb", "---")
        if isinstance(vram_mb, (int, float)):
            vram_mb = f"{vram_mb:.1f}M"
            
        rows.append({
            "mode": mode,
            "dataset": dataset,
            "method": method_str,
            "acc": acc,
            "trainable_m": trainable_m,
            "pct": trainable_pct,
            "time_m": time_m,
            "ckpt": ckpt_mb,
            "vram": vram_mb
        })
        
    # Sort for consistent ordering
    rows.sort(key=lambda x: (x["dataset"], x["mode"]))

    # Generate LaTeX
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
    latex.append("  Method & Dataset & Accuracy & Params & \\% & Time & Ckpt Size & Peak VRAM \\\\")
    latex.append("  \\midrule")
    
    for r in rows:
        dataset_upper = r["dataset"].upper()
        latex.append(f"  {r['method']:<18} & {dataset_upper:<6} & {r['acc']:>4.1f}\\% & {r['trainable_m']:>4.1f}M & {r['pct']:>5.2f}\\% & {r['time_m']:>4.1f}m & {r['ckpt']:>8} & {r['vram']:>8} \\\\")
        
    latex.append("  \\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\caption{Automated training results comparison.}")
    latex.append("\\end{table}")
    latex.append("")
    latex.append("\\end{document}")
    
    print("\n" + "\n".join(latex) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LaTeX table from JSON results.")
    parser.add_argument("--results_dir", type=str, default="./results", help="Directory containing JSON result files.")
    args = parser.parse_args()
    
    generate_latex_table(args.results_dir)
