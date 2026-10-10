import os, random
import numpy as np

SEED = 42            
IMG_SIZE = 128       
BATCH_SIZE = 32
LR = 0.001
EPOCHS = 20          
PATIENCE = 5         
DROPOUT = 0.5

DATA_DIR = "data/dataset-resized"   
RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
MODELS_DIR = "models"

ALL_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
CLASSES = ALL_CLASSES if os.environ.get("INCLUDE_TRASH") == "1" else ALL_CLASSES[:5]
NUM_CLASSES = len(CLASSES)

MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)

for d in (RESULTS_DIR, PLOTS_DIR, MODELS_DIR):
    os.makedirs(d, exist_ok=True)


def set_seed(seed=SEED):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
