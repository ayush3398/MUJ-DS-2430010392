"""CO4 - Compare HOG+SVM, MLP (flattened pixels) and our CNN on the SAME split and the SAME test images."""
import pandas as pd, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import *
from dataset_analysis import load_split
from preprocessing import get_loader
from model import FlatMLP, count_parameters
from train import train_model, FINAL_MODEL, DEVICE
from evaluate import predict_loader, compute_metrics, load_cnn
from hog_svm import run_hog_svm

df = load_split()
rows = []

print("\n--- HOG + SVM ---")
r = run_hog_svm(df)
rows.append({"model": "HOG + SVM", "accuracy": r["accuracy"], "precision_macro": r["precision"],
             "recall_macro": r["recall"], "f1_macro": r["f1"], "trainable_params": "n/a", "note": r["note"]})

print("\n--- MLP on flattened pixels ---")
set_seed(SEED)
mlp = FlatMLP()
train_model("mlp", mlp, get_loader("train"), get_loader("val"))     # same optimizer / lr / batch size / early stopping
test_loader = get_loader("test")
m = compute_metrics(*predict_loader(mlp, test_loader)[:2])
rows.append({"model": "MLP (flatten)", "accuracy": m["accuracy"], "precision_macro": m["precision"],
             "recall_macro": m["recall"], "f1_macro": m["f1"], "trainable_params": count_parameters(mlp),
             "note": "no augmentation, no convolution"})

print("\n--- CNN (final model) ---")
cnn = load_cnn(FINAL_MODEL)
m = compute_metrics(*predict_loader(cnn, test_loader)[:2])
rows.append({"model": "CNN (aug + dropout)", "accuracy": m["accuracy"], "precision_macro": m["precision"],
             "recall_macro": m["recall"], "f1_macro": m["f1"], "trainable_params": count_parameters(cnn),
             "note": "final model from train.py"})

out = pd.DataFrame(rows)
out[["accuracy", "precision_macro", "recall_macro", "f1_macro"]] = out[["accuracy", "precision_macro", "recall_macro", "f1_macro"]].round(4)
out.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)
print("\n", out.to_string(index=False))

ax = out.set_index("model")[["accuracy", "precision_macro", "recall_macro", "f1_macro"]].plot(kind="bar", figsize=(9, 5))
ax.set_ylim(0, 1); ax.set_ylabel("Score"); ax.set_title("Model comparison on the same test set"); plt.xticks(rotation=0)
plt.legend(loc="lower right"); plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, "model_comparison.png"), dpi=150)
print("Saved results/model_comparison.csv and plots/model_comparison.png")
