import os, json, hashlib
from collections import Counter
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from config import *

SPLIT_CSV = os.path.join(RESULTS_DIR, "split.csv")


def _md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def list_images():
    rows = []
    for label, cls in enumerate(CLASSES):
        folder = os.path.join(DATA_DIR, cls)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"{folder} not found. Run: python download_data.py")
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                rows.append((os.path.join(folder, f), cls, label))
    df = pd.DataFrame(rows, columns=["path", "class", "label"])
    df["hash"] = df["path"].map(_md5)
    n_before = len(df)
    df = df.drop_duplicates(subset="hash", keep="first").drop(columns="hash").reset_index(drop=True)
    print(f"Exact duplicate images removed before splitting: {n_before - len(df)}")
    df.attrs["duplicates_removed"] = n_before - len(df)
    return df


def make_split(df):
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label"], random_state=SEED)
    return pd.concat([train_df.assign(split="train"), val_df.assign(split="val"),
                      test_df.assign(split="test")]).reset_index(drop=True)


def load_split():
    if os.path.exists(SPLIT_CSV):
        df = pd.read_csv(SPLIT_CSV)
        if set(df["class"]) == set(CLASSES):
            return df
    df = make_split(list_images())
    df.to_csv(SPLIT_CSV, index=False)
    return df


def check_leakage(df):
    """No file may be in two splits, and no byte-identical image may be in two splits."""
    same_path = int((df.groupby("path")["split"].nunique() > 1).sum())
    hashes = df.assign(h=df["path"].map(_md5)).groupby("h")["split"].nunique()
    return {"files_in_more_than_one_split": same_path,
            "identical_image_groups_across_splits": int((hashes > 1).sum()),
            "duplicate_groups_total": int((df.assign(h=df["path"].map(_md5)).groupby("h").size() > 1).sum())}


def analyze_dataset():
    df = load_split()
    sizes = Counter(Image.open(p).size for p in df["path"])
    per_class = df["class"].value_counts().reindex(CLASSES)
    split_tab = df.groupby(["split", "class"]).size().unstack(fill_value=0).reindex(index=["train", "val", "test"], columns=CLASSES)
    stats = {
        "dataset": "TrashNet (dataset-resized), github.com/garythung/trashnet",
        "total_images": int(len(df)),
        "num_classes": int(NUM_CLASSES),
        "classes": CLASSES,
        "images_per_class": {c: int(per_class[c]) for c in CLASSES},
        "class_percent": {c: round(float(100 * per_class[c] / len(df)), 2) for c in CLASSES},
        "imbalance_ratio_max_over_min": round(float(per_class.max() / per_class.min()), 2),
        "original_image_sizes_WxH": {f"{w}x{h}": int(n) for (w, h), n in sizes.items()},
        "model_input_size": f"{IMG_SIZE}x{IMG_SIZE}x3",
        "split_counts": {s: int(split_tab.loc[s].sum()) for s in split_tab.index},
        "split_per_class": {s: {c: int(split_tab.loc[s, c]) for c in CLASSES} for s in split_tab.index},
        "raw_images_before_dedup": int(sum(len(os.listdir(os.path.join(DATA_DIR, c))) for c in CLASSES)),
        "leakage_check": check_leakage(df),
    }
    with open(os.path.join(RESULTS_DIR, "dataset_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print("=" * 60, "\nDATASET STATISTICS (computed from files)\n" + "=" * 60)
    print("Total images :", stats["total_images"])
    print("Classes      :", stats["num_classes"], CLASSES)
    print("Images/class :", stats["images_per_class"])
    print("Class %      :", stats["class_percent"])
    print("Imbalance    : largest/smallest class =", stats["imbalance_ratio_max_over_min"])
    print("Original size (WxH):", stats["original_image_sizes_WxH"], "-> resized to", stats["model_input_size"])
    print("Split counts :", stats["split_counts"])
    print(split_tab)
    print("Leakage check:", stats["leakage_check"])

    # Plot 1: class distribution (whole dataset)
    plt.figure(figsize=(7, 4))
    bars = plt.bar(CLASSES, [per_class[c] for c in CLASSES], color="steelblue")
    for b in bars:
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 3, int(b.get_height()), ha="center")
    plt.title("Class distribution (all images)"); plt.ylabel("Number of images"); plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "class_distribution.png"), dpi=150); plt.close()

    # Plot 2: class distribution inside each split
    split_tab.T.plot(kind="bar", figsize=(8, 4))
    plt.title("Class distribution per split"); plt.ylabel("Number of images"); plt.xticks(rotation=0); plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "split_distribution.png"), dpi=150); plt.close()

    # Plot 3: 5 sample images from every class
    rng = np.random.RandomState(SEED)
    fig, axes = plt.subplots(NUM_CLASSES, 5, figsize=(10, 2 * NUM_CLASSES))
    for i, c in enumerate(CLASSES):
        paths = df[df["class"] == c]["path"].values
        for j, p in enumerate(rng.choice(paths, 5, replace=False)):
            axes[i, j].imshow(Image.open(p).convert("RGB")); axes[i, j].axis("off")
            if j == 0:
                axes[i, j].set_title(c, loc="left", fontsize=11, fontweight="bold")
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, "sample_images.png"), dpi=150); plt.close()
    print("Saved results/dataset_stats.json and plots.")
    return stats


if __name__ == "__main__":
    analyze_dataset()
