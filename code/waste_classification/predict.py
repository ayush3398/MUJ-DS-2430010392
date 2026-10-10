import sys, torch
from PIL import Image
from config import *
from preprocessing import get_transform
from evaluate import load_cnn
from train import FINAL_MODEL, DEVICE

if len(sys.argv) < 2:
    sys.exit("Usage: python predict.py image1.jpg [image2.jpg ...]")
model = load_cnn(FINAL_MODEL).eval()
tf = get_transform(augment=False)                      # same preprocessing as validation/test
for path in sys.argv[1:]:
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    with torch.no_grad():
        probs = torch.softmax(model(tf(img).unsqueeze(0).to(DEVICE)), dim=1)[0].cpu()
    print(f"\n{path}\n  Predicted: {CLASSES[int(probs.argmax())]}  ({probs.max():.1%})")
    for c, p in sorted(zip(CLASSES, probs.tolist()), key=lambda t: -t[1]):
        print(f"    {c:10s} {p:.1%}")
