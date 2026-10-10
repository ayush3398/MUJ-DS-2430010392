import torch.nn as nn
from config import *


class WasteCNN(nn.Module):

    def __init__(self, num_classes=NUM_CLASSES, dropout=DROPOUT):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  nn.ReLU(), nn.MaxPool2d(2),   # 32 x 64 x 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 64 x 32 x 32
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 128 x 16 x 16
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                                # 128*16*16 = 32768 numbers
            nn.Linear(128 * 16 * 16, 128), nn.ReLU(),
            nn.Dropout(dropout),                         # dropout=0.0 means "no dropout" (used in ablation)
            nn.Linear(128, num_classes),                 # raw scores (logits)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class FlatMLP(nn.Module):
    

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(3 * IMG_SIZE * IMG_SIZE, 256), nn.ReLU(),
                                 nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, num_classes))

    def forward(self, x):
        return self.net(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def draw_architecture(path=os.path.join(PLOTS_DIR, "cnn_architecture.png")):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    layers = [("Input\n3x128x128", "#dddddd"), ("Conv 3x3\n32 + ReLU\n32x128x128", "#9ecae1"),
              ("MaxPool 2x2\n32x64x64", "#fdae6b"), ("Conv 3x3\n64 + ReLU\n64x64x64", "#9ecae1"),
              ("MaxPool 2x2\n64x32x32", "#fdae6b"), ("Conv 3x3\n128 + ReLU\n128x32x32", "#9ecae1"),
              ("MaxPool 2x2\n128x16x16", "#fdae6b"), ("Flatten\n32768", "#c7e9c0"),
              ("Dense 128\n+ ReLU", "#bcbddc"), ("Dropout\n(p=0.5)", "#fcbba1"),
              (f"Dense {NUM_CLASSES}\n+ Softmax", "#dadaeb")]
    fig, ax = plt.subplots(figsize=(20, 3))
    w, gap = 1.5, 0.35
    for i, (txt, col) in enumerate(layers):
        x = i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0), w, 1.6, boxstyle="round,pad=0.05", fc=col, ec="black"))
        ax.text(x + w / 2, 0.8, txt, ha="center", va="center", fontsize=9)
        if i < len(layers) - 1:
            ax.annotate("", xy=(x + w + gap, 0.8), xytext=(x + w, 0.8), arrowprops=dict(arrowstyle="->"))
    ax.set_xlim(-0.2, len(layers) * (w + gap)); ax.set_ylim(-0.2, 1.8); ax.axis("off")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


if __name__ == "__main__":
    import torch
    cnn = WasteCNN()
    print(cnn)
    print("Trainable parameters - CNN:", count_parameters(cnn), "| MLP:", count_parameters(FlatMLP()))
    print("Output shape for a batch of 4:", cnn(torch.randn(4, 3, IMG_SIZE, IMG_SIZE)).shape)
    draw_architecture()
    print("Saved results/plots/cnn_architecture.png")
