import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from config import *
from dataset_analysis import load_split

_CACHE = {}  


def load_image(path):
    if path not in _CACHE:
        _CACHE[path] = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return _CACHE[path]


def get_transform(augment):
    base = [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    if augment:
        aug = [transforms.RandomHorizontalFlip(0.5),
               transforms.RandomRotation(15),
               transforms.ColorJitter(brightness=0.2, contrast=0.2)]
        return transforms.Compose(aug + base)
    return transforms.Compose(base)


class WasteDataset(Dataset):
    def __init__(self, df, augment=False):
        self.paths = df["path"].tolist()
        self.labels = df["label"].tolist()
        self.images = [load_image(p) for p in self.paths]   # resized PIL images (kept for plotting too)
        self.transform = get_transform(augment)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.transform(self.images[i]), self.labels[i]


def get_loader(split, augment=False):
    """split = 'train' | 'val' | 'test'. Only the train loader shuffles."""
    df = load_split()
    ds = WasteDataset(df[df["split"] == split], augment=augment)
    g = torch.Generator()
    g.manual_seed(SEED)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=(split == "train"), generator=g, num_workers=0)


if __name__ == "__main__":
    for s in ("train", "val", "test"):
        print(s, len(get_loader(s).dataset), "images")
    x, y = next(iter(get_loader("train", augment=True)))
    print("One batch:", x.shape, y.shape, "| value range:", float(x.min()), "to", float(x.max()))
