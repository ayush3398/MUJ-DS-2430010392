import numpy as np
from PIL import Image
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from config import *


def hog_features(paths):
    feats = []
    for p in paths:
        gray = np.asarray(Image.open(p).convert("L").resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32) / 255.0
        feats.append(hog(gray, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2)))
    return np.array(feats)


def run_hog_svm(df):
    tr, va, te = (df[df["split"] == s] for s in ("train", "val", "test"))
    Xtr, Xva, Xte = hog_features(tr["path"]), hog_features(va["path"]), hog_features(te["path"])
    # choose C on the VALIDATION set only (test set is never used for choices)
    best_c, best_acc = None, -1
    for c in (0.1, 1, 10):
        acc = accuracy_score(va["label"], SVC(kernel="rbf", C=c).fit(Xtr, tr["label"]).predict(Xva))
        print(f"  SVM C={c}: val acc={acc:.4f}")
        if acc > best_acc:
            best_c, best_acc = c, acc
    svm = SVC(kernel="rbf", C=best_c).fit(Xtr, tr["label"])
    pred = svm.predict(Xte)
    p, r, f1, _ = precision_recall_fscore_support(te["label"], pred, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(te["label"], pred), "precision": p, "recall": r, "f1": f1,
            "note": f"HOG({Xtr.shape[1]} features)+RBF-SVM, C={best_c}"}
