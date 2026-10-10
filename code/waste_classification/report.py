import json
import pandas as pd
from config import *


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def delta_sentence(a, b, what):
    d = (b - a) * 100
    if abs(d) < 1:
        verdict = "made almost no difference (under 1 percentage point, which is within normal run-to-run noise)"
    else:
        verdict = f"{'improved' if d > 0 else 'reduced'} test accuracy by {abs(d):.1f} percentage points"
    return f"- {what} {verdict} ({a:.4f} -> {b:.4f})."


stats = json.load(open(os.path.join(RESULTS_DIR, "dataset_stats.json")))
out = ["# Results (auto-generated from the actual experiments)\n", "## Dataset statistics\n",
       f"- Source: {stats['dataset']}", f"- Images used: {stats['total_images']} (raw images in chosen folders: {stats['raw_images_before_dedup']}; exact duplicates removed before splitting)",
       f"- Classes ({stats['num_classes']}): {', '.join(stats['classes'])}",
       f"- Original size (WxH): {stats['original_image_sizes_WxH']} -> resized to {stats['model_input_size']}",
       f"- Images per class: {stats['images_per_class']}", f"- Class percentages: {stats['class_percent']}",
       f"- Imbalance ratio (largest/smallest class): {stats['imbalance_ratio_max_over_min']}",
       f"- Split counts: {stats['split_counts']}", f"- Leakage check: {stats['leakage_check']}\n"]

if os.path.exists(os.path.join(RESULTS_DIR, "metrics.json")):
    m = json.load(open(os.path.join(RESULTS_DIR, "metrics.json")))
    out += ["## Final CNN (augmentation + dropout) on the test set\n",
            f"Accuracy {m['accuracy']:.4f} | Precision (macro) {m['precision']:.4f} | Recall (macro) {m['recall']:.4f} | F1 (macro) {m['f1']:.4f} | test images: {m['test_images']}\n",
            "```\n" + open(os.path.join(RESULTS_DIR, "classification_report.txt")).read() + "```\n"]

ab_path = os.path.join(RESULTS_DIR, "ablation_results.csv")
if os.path.exists(ab_path):
    ab = pd.read_csv(ab_path)
    out += ["## Ablation study (CO5)\n", md_table(ab), "\n**Automatic reading of the table** (single run, one seed, so small differences are not conclusive):\n"]
    acc = dict(zip(ab["experiment"], ab["test_accuracy"]))
    if "baseline" in acc and "augmentation" in acc:
        out.append(delta_sentence(acc["baseline"], acc["augmentation"], "Adding augmentation"))
    if "augmentation" in acc and "augmentation_dropout" in acc:
        out.append(delta_sentence(acc["augmentation"], acc["augmentation_dropout"], "Adding dropout on top of augmentation"))
    if "baseline" in acc and "augmentation_dropout" in acc:
        out.append(delta_sentence(acc["baseline"], acc["augmentation_dropout"], "Both together versus the baseline"))
    out.append("")

cmp_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
if os.path.exists(cmp_path):
    cmp_ = pd.read_csv(cmp_path)
    best = cmp_.loc[cmp_["accuracy"].idxmax()]
    out += ["## Model comparison (CO4)\n", md_table(cmp_), f"\nBest test accuracy: **{best['model']}** ({best['accuracy']:.4f}).\n"]

open(os.path.join(RESULTS_DIR, "RESULTS.md"), "w").write("\n".join(out))
print("\n".join(out))
