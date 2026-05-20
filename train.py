import copy
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from src.helper import create_next_run_dir, init_model, load_checkpoint, seed_everything
from src.transforms import GaussianBlur
from tqdm import tqdm

from src.models.tec import TwoEncoderClassifier, evaluate, predict, run_epoch
from src.utils.config_loader import load_config
from src.utils.data import get_dataloaders, get_tfms
from src.utils.evaluation import ClassificationEvaluator
from src.utils.loss import get_class_weights
from src.utils.train import EarlyStopping
from src.utils.visualize import plot_confusion_matrix

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ---- CONFIG ----
cfg = load_config("configs/train.yaml")
jepa_path = cfg["train"]["jepa_checkpoint"]
data_dir = cfg["data"]["data_dir"]
img_size = cfg["data"]["img_size"]

batch_size = cfg["train"]["batch_size"]
num_epochs = cfg["train"]["num_epochs"]
lr = cfg["train"]["lr"]
num_workers = cfg["train"]["num_workers"]
patience = cfg["train"]["patience"]
precision = cfg["train"]["precision"]
wd = cfg["train"]["weight_decay"]
seeds = cfg["seeds"]

patch_size = cfg["train"]["jepa"]["patch_size"]
model_name = cfg["train"]["jepa"]["model_name"]
crop_size = img_size
pred_depth = cfg["train"]["jepa"]["pred_depth"]
pred_emb_dim = cfg["train"]["jepa"]["pred_emb_dim"]
results_path = Path(cfg["train"]["results_path"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_tfms, val_tfms = get_tfms(img_size)
print("train_tfms:", train_tfms)


def run_seed(seed: int):

    seed_everything(seed)
    run_dir = create_next_run_dir(results_path, seed=seed)

    # ---- DATA ----
    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = get_dataloaders(
        data_dir=data_dir,
        seed=seed,
        batch_size=batch_size,
        num_workers=num_workers,
        train_tfms=train_tfms,
        val_tfms=val_tfms,
    )

    # ---- MODEL ----
    encoder, predictor = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name,
    )
    target_encoder = copy.deepcopy(encoder)

    encoder, predictor, target_encoder = load_checkpoint(
        device=device,
        r_path=jepa_path,
        encoder=encoder,
        predictor=predictor,
        target_encoder=target_encoder,
    )

    model = TwoEncoderClassifier(encoder, 768, 10)
    model = model.to(device)

    # ---- LOSS / OPTIM ----
    class_weights = get_class_weights(train_ds).to(device)  # class-balance loss
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # ---- EVAL ----
    evaluator = ClassificationEvaluator()

    # ---- TRAIN LOOP ----
    early_stopper = EarlyStopping(patience=patience)
    best_model = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0

    losses = {}
    epochs_no_improve = 0

    for epoch in tqdm(range(num_epochs)):
        t0 = time.time()
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            train=True,
        )
        val_loss, val_acc = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            train=False,
        )

        losses[epoch] = {"train_loss": train_loss, "val_loss": val_loss}

        metrics, cm = evaluator.evaluate(
            model=model,
            loader=val_loader,
            averages=["macro", "micro"],
        )

        val_acc = metrics["accuracy"]
        val_prec = metrics["precision_micro"]
        val_rec = metrics["recall_micro"]
        val_f1 = metrics["f1_micro"]
        val_prec_macro = metrics["precision_macro"]
        val_rec_macro = metrics["recall_macro"]
        val_f1_macro = metrics["f1_macro"]

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            print(f"New best val acc: {best_val_acc:.{precision}f} - saving model.")
        else:
            epochs_no_improve += 1
        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"train loss {train_loss:.{precision}f} acc {train_acc:.{precision}f} | "
            f"val loss {val_loss:.{precision}f} acc {val_acc:.{precision}f} | "
            f"P {val_prec:.{precision}f} R {val_rec:.{precision}f} F1 {val_f1:.{precision}f} | "
            f"macro P {val_prec_macro:.{precision}f} R {val_rec_macro:.{precision}f} F1 {val_f1_macro:.{precision}f} | "
            f"{time.time()-t0:.4f}s"
        )

        if early_stopper.should_stop and epoch >= 10:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Loss plotting
    plt.figure(figsize=(10, 5))
    plt.plot([losses[i]["train_loss"] for i in range(num_epochs)], label="Train Loss")
    plt.plot([losses[i]["val_loss"] for i in range(num_epochs)], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Val Loss")
    plt.savefig(run_dir / "loss_plot.png")

    # Load best model
    model.load_state_dict(best_model)

    # ---- EVALUATION ----
    test_loss, test_acc = run_epoch(
        model,
        test_loader,
        criterion,
        optimizer,
        train=False,
    )
    logger.info(f"Best val acc: {best_val_acc:.{precision}f}")
    logger.info(
        f"Test loss: {test_loss:.{precision}f} | Test acc: {test_acc:.{precision}f}"
    )

    torch.save(
        {"model_state": model.state_dict(), "classes": train_ds.classes},
        run_dir / "model.pth",
    )
    logger.info(f"Saved to {run_dir}")

    # Save detailed classification report
    metrics, cm = evaluator.evaluate(
        model,
        test_loader,
        results_path=run_dir,
        save_confusion=True,
    )

    plot_confusion_matrix(
        cm,
        class_names=test_ds.classes,
        normalize=False,
        title="Confusion Matrix - Test Set",
        save_path=run_dir / "cm.png",
    )

    logger.info("Evaluation complete.")


for seed in seeds:
    logger.info(f"=== Running seed {seed} ===")
    run_seed(seed)
