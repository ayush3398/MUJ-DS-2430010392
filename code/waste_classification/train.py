import argparse, json, time, copy
import torch, torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import *
from preprocessing import get_loader
from model import WasteCNN, count_parameters



DEVICE = get_device()
EXPERIMENTS = {
    "baseline":             dict(augment=False, dropout=0.0),      # Exp 1
    "augmentation":         dict(augment=True,  dropout=0.0),      # Exp 2
    "augmentation_dropout": dict(augment=True,  dropout=DROPOUT),  # Exp 3 = final model
}
FINAL_MODEL = "augmentation_dropout"


def run_epoch(model, loader, criterion, optimizer=None):
    """One pass over the data. With an optimizer -> training; without -> evaluation only."""
    training = optimizer is not None
    model.train(training)                       
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):     
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)                     
            loss = criterion(out, y)            
            if training:
                optimizer.zero_grad()           
                loss.backward()                 
                optimizer.step()                
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def train_model(name, model, train_loader, val_loader, epochs=EPOCHS):
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    hist = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc")}
    best_val, best_state, best_epoch, bad = float("inf"), None, 0, 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        tl, ta = run_epoch(model, train_loader, criterion, optimizer)
        vl, va = run_epoch(model, val_loader, criterion)
        for k, v in zip(hist, (tl, ta, vl, va)):
            hist[k].append(v)
        print(f"[{name}] epoch {ep:2d}/{epochs} | train loss {tl:.4f} acc {ta:.4f} | val loss {vl:.4f} acc {va:.4f}")
        if vl < best_val:                       
            best_val, best_epoch, bad = vl, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"[{name}] early stopping at epoch {ep} (best epoch {best_epoch})")
                break
    model.load_state_dict(best_state)           
    hist.update(best_epoch=best_epoch, epochs_run=len(hist["val_loss"]), best_val_loss=best_val,
                best_val_acc=hist["val_acc"][best_epoch - 1], train_seconds=round(time.time() - t0, 1),
                trainable_params=count_parameters(model))
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"{name}.pt"))
    with open(os.path.join(RESULTS_DIR, f"history_{name}.json"), "w") as f:
        json.dump(hist, f, indent=2)
    plot_history(hist, name)
    return hist


def plot_history(hist, name):
    epochs = range(1, len(hist["train_loss"]) + 1)
    for key, title, ylabel in (("train_acc", "Training accuracy", "Accuracy"), ("val_acc", "Validation accuracy", "Accuracy"),
                               ("train_loss", "Training loss", "Loss"), ("val_loss", "Validation loss", "Loss")):
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, hist[key], marker="o")
        plt.axvline(hist["best_epoch"], color="gray", linestyle="--", label="best epoch")
        plt.title(f"{title} vs epoch ({name})"); plt.xlabel("Epoch"); plt.ylabel(ylabel); plt.legend(); plt.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, f"{name}_{key}.png"), dpi=150); plt.close()
    # combined curves for the final model / every experiment (train vs val on the same axes)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(epochs, hist["train_acc"], label="train"); ax[0].plot(epochs, hist["val_acc"], label="validation")
    ax[0].set_title("Accuracy"); ax[0].set_xlabel("Epoch"); ax[0].legend()
    ax[1].plot(epochs, hist["train_loss"], label="train"); ax[1].plot(epochs, hist["val_loss"], label="validation")
    ax[1].set_title("Loss"); ax[1].set_xlabel("Epoch"); ax[1].legend()
    plt.suptitle(name); plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, f"{name}_curves.png"), dpi=150); plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all", choices=["all"] + list(EXPERIMENTS))
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()
    print("Device:", DEVICE, "| classes:", CLASSES)
    val_loader = get_loader("val")                       
    for name, cfg in EXPERIMENTS.items():
        if args.exp not in ("all", name):
            continue
        set_seed(SEED)                                  
        train_loader = get_loader("train", augment=cfg["augment"])
        model = WasteCNN(dropout=cfg["dropout"])
        print(f"\n=== {name}: augment={cfg['augment']} dropout={cfg['dropout']} | params={count_parameters(model):,} ===")
        train_model(name, model, train_loader, val_loader, args.epochs)
    print("\nTraining finished. Next: python evaluate.py")
