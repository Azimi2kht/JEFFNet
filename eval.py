import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
from src.helper import init_model, seed_everything

from src.models.tec import TwoEncoderClassifier
from src.utils.config_loader import load_config
from src.utils.data import get_dataloaders, get_tfms
from src.utils.evaluation import ClassificationEvaluator

# ---- CONFIG ----
cfg = load_config("configs/eval.yaml")

# data
data_dir = cfg["data"]["data_dir"]
img_size = cfg["data"]["img_size"]

# model paths
model_path = cfg["model_path"]

# training params
batch_size = cfg["train"]["batch_size"]
num_workers = cfg["train"]["num_workers"]
precision = cfg["train"]["precision"]
seed = cfg["train"]["seed"]

# JEPA params
patch_size = cfg["jepa"]["patch_size"]
model_name = cfg["jepa"]["model_name"]
pred_depth = cfg["jepa"]["pred_depth"]
pred_emb_dim = cfg["jepa"]["pred_emb_dim"]

crop_size = img_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_tfms, val_tfms = get_tfms(img_size)
seed_everything(seed)

# ---- DATA ----
train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = get_dataloaders(
    data_dir=data_dir,
    seed=seed,
    batch_size=batch_size,
    num_workers=num_workers,
    train_tfms=train_tfms,
    val_tfms=val_tfms,
)

# ---- MODEL INIT ----
encoder, predictor = init_model(
    device=device,
    patch_size=patch_size,
    crop_size=crop_size,
    pred_depth=pred_depth,
    pred_emb_dim=pred_emb_dim,
    model_name=model_name,
)
del predictor

# ---- LOAD WEIGHTS ----
model = TwoEncoderClassifier(encoder, 768, 10)
model.load_state_dict(torch.load(model_path)["model_state"])
model = model.to(device)

# ---- EVAL ----
evaluator = ClassificationEvaluator()
metrics, cm = evaluator.evaluate(
    model,
    test_loader,
    save_confusion=True,
)
print("Metrics:", metrics)
print("Confusion Matrix:", cm)
