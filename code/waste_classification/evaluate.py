import json
import numpy as np, pandas as pd, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report)
from config import *
from preprocessing import get_loader
from model import WasteCNN
from train import EXPERIMENTS, FINAL_MODEL, DEVICE


@torch.no_grad()
def predict_loader(model, loader):
    model.eval()                                    # dropout OFF at test time
    probs = []
    for x, _ in loader:
        probs.append(torch.softmax(model(x.to(DEVICE)), dim=1).cpu())   # softmax -> class probabilities
    probs = torch.cat(probs).numpy()
    return np.array(loader.dataset.labels), probs.argmax(1), probs


def compute_metrics(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred), "precision": p, "recall": r, "f1": f1}   # macro = average over classes


def load_cnn(name):
    model = WasteCNN(dropout=EXPERIMENTS[name]["dropout"]).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"{name}.pt"), map_location=DEVICE))
    return model


def plot_confusion(cm, path, normalize=False):
    data = cm / cm.sum(axis=1, keepdims=True) if normalize else cm
    plt.figure(figsize=(6, 5))
    plt.imshow(data, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(NUM_CLASSES), CLASSES, rotation=45); plt.yticks(range(NUM_CLASSES), CLASSES)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            plt.text(j, i, f"{data[i, j]:.2f}" if normalize else int(data[i, j]), ha="center", va="center",
                     color="white" if data[i, j] > data.max() / 2 else "black")
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion matrix" + (" (row-normalised)" if normalize else ""))
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_predictions(ds, y_true, y_pred, probs, path, title, idx):
    cols = 4
    rows = int(np.ceil(len(idx) / cols))
    plt.figure(figsize=(3 * cols, 3.2 * rows))
    for k, i in enumerate(idx):
        plt.subplot(rows, cols, k + 1)
        plt.imshow(ds.images[i]); plt.axis("off")
        ok = y_true[i] == y_pred[i]
        plt.title(f"Actual: {CLASSES[y_true[i]]}\nPred: {CLASSES[y_pred[i]]} ({probs[i].max():.0%})",
                  color="green" if ok else "red", fontsize=9)
    plt.suptitle(title); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


if __name__ == "__main__":
    test_loader = get_loader("test")                       # no augmentation
    rows = []
    for name, cfg in EXPERIMENTS.items():
        if not os.path.exists(os.path.join(MODELS_DIR, f"{name}.pt")):
            print("Skipping", name, "(not trained yet)"); continue
        y_true, y_pred, probs = predict_loader(load_cnn(name), test_loader)
        m = compute_metrics(y_true, y_pred)
        hist = json.load(open(os.path.join(RESULTS_DIR, f"history_{name}.json")))
        rows.append({"experiment": name, "augmentation": cfg["augment"], "dropout": cfg["dropout"],
                     "test_accuracy": m["accuracy"], "precision_macro": m["precision"], "recall_macro": m["recall"],
                     "f1_macro": m["f1"], "best_epoch": hist["best_epoch"], "best_val_accuracy": hist["best_val_acc"]})
        print(f"{name:22s} test acc {m['accuracy']:.4f} | P {m['precision']:.4f} R {m['recall']:.4f} F1 {m['f1']:.4f}")
        if name == FINAL_MODEL:
            final = (y_true, y_pred, probs, m)
    pd.DataFrame(rows).round(4).to_csv(os.path.join(RESULTS_DIR, "ablation_results.csv"), index=False)

    # ---- detailed outputs for the FINAL model ----
    y_true, y_pred, probs, m = final
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion(cm, os.path.join(PLOTS_DIR, "confusion_matrix.png"))
    plot_confusion(cm, os.path.join(PLOTS_DIR, "confusion_matrix_normalized.png"), normalize=True)
    report = classification_report(y_true, y_pred, target_names=CLASSES, digits=4, zero_division=0)
    open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w").write(report)
    print("\nClassification report (final model):\n", report)
    json.dump({"model": FINAL_MODEL, **{k: float(v) for k, v in m.items()}, "test_images": int(len(y_true)),
               "confusion_matrix": cm.tolist()}, open(os.path.join(RESULTS_DIR, "metrics.json"), "w"), indent=2)
    ds = test_loader.dataset
    pd.DataFrame({"path": ds.paths, "actual": [CLASSES[i] for i in y_true], "predicted": [CLASSES[i] for i in y_pred],
                  "confidence": probs.max(1).round(4), "correct": y_true == y_pred}).to_csv(
        os.path.join(RESULTS_DIR, "predictions.csv"), index=False)
    rng = np.random.RandomState(SEED)
    plot_predictions(ds, y_true, y_pred, probs, os.path.join(PLOTS_DIR, "example_predictions.png"),
                     "Example test predictions (random, green = correct, red = wrong)", rng.choice(len(y_true), 12, replace=False))
    wrong = np.where(y_true != y_pred)[0]
    if len(wrong):
        plot_predictions(ds, y_true, y_pred, probs, os.path.join(PLOTS_DIR, "misclassified_examples.png"),
                         "Misclassified test images", wrong[:12])
    print("Saved all evaluation outputs in results/")
