import os

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torchvision import models

device = "cuda" if torch.cuda.is_available() else "cpu"


class EfficientNetV2SFeatures(nn.Module):
    """EfficientNetV2-S backbone that outputs a 1D feature vector per image."""

    def __init__(
        self,
        weights=models.EfficientNet_V2_S_Weights.DEFAULT,
        train_backbone: bool = True,
    ):
        super().__init__()
        m = models.efficientnet_v2_s(weights=weights)

        # same idea as EfficientNet-B0
        self.features = m.features
        self.avgpool = m.avgpool

        # classifier is Sequential(Dropout, Linear) in torchvision
        self.out_dim = m.classifier[1].in_features

        if not train_backbone:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x):
        x = self.features(x)  # [B, C, H, W]
        x = self.avgpool(x)  # [B, C, 1, 1]
        x = torch.flatten(x, 1)  # [B, C]
        return x


class TwoEncoderClassifier(nn.Module):
    def __init__(
        self,
        jepa_encoder: nn.Module,
        jepa_emb_dim: int,
        num_classes: int,
        weights=models.EfficientNet_V2_S_Weights.DEFAULT,
        fusion_hidden: int = 512,
        dropout: float = 0.2,
        train_backbone: bool = True,
    ):
        super().__init__()
        self.eff = EfficientNetV2SFeatures(
            weights=weights, train_backbone=train_backbone
        )
        self.jepa = jepa_encoder

        fused_dim = self.eff.out_dim + jepa_emb_dim

        self.head = nn.Sequential(
            nn.Linear(fused_dim, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, num_classes),
        )

    def extract_modal_features(self, img):
        f1 = self.eff(img)                 # [B, eff_dim]
        f2 = self.jepa(img).mean(dim=1)    # [B, jepa_dim]
        return f1, f2

    def extract_fused_features(self, img):
        f1, f2 = self.extract_modal_features(img)
        fused = torch.cat([f1, f2], dim=1)
        return fused

    def extract_penultimate_features(self, img):
        fused = self.extract_fused_features(img)
        x = self.head[0](fused)   # first Linear
        x = self.head[1](x)       # ReLU
        return x

    def forward(self, img):
        fused = self.extract_fused_features(img)
        return self.head(fused)


def predict(model, loader, device=device):
    """
    Returns:
        y_pred (np.ndarray): predicted class index, shape [N]
        y_true (np.ndarray): true class index, shape [N]
        y_prob (np.ndarray): predicted probabilities, shape [N, C]
    """
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            logits = model(imgs)  # [B, C]
            probs = torch.softmax(logits, dim=1)  # [B, C]
            preds = torch.argmax(probs, dim=1)  # [B]

            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    y_prob = np.concatenate(all_probs, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    return y_pred, y_true, y_prob


def evaluate(model, loader, classes, device=device, save_path=None, log=False):
    """
    Computes metrics for multi-class classification:
      - accuracy
      - macro precision/recall/f1 (and micro too, if you want)
      - macro AUC (OvR), matching the paper's "mean over classes" idea
      - MCC
      - classification report + confusion matrix
    """
    y_pred, y_true, y_prob = predict(model, loader, device)
    num_classes = len(classes)

    acc = accuracy_score(y_true, y_pred)
    pre_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    pre_micro = precision_score(y_true, y_pred, average="micro", zero_division=0)
    rec_micro = recall_score(y_true, y_pred, average="micro", zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)

    y_true_oh = np.eye(num_classes)[y_true]  # one-hot
    auc_macro = roc_auc_score(y_true_oh, y_prob, average="macro", multi_class="ovr")
    auc_micro = roc_auc_score(y_true_oh, y_prob, average="micro", multi_class="ovr")

    mcc = matthews_corrcoef(y_true, y_pred)

    report = classification_report(
        y_true, y_pred, target_names=classes, digits=4, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "accuracy": acc,
        "precision_macro": pre_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "auc_macro_ovr": auc_macro,
        "precision_micro": pre_micro,
        "recall_micro": rec_micro,
        "f1_micro": f1_micro,
        "auc_micro_ovr": auc_micro,
        "mcc": mcc,
    }

    if log:
        print("Metrics:", {k: round(v, 6) for k, v in metrics.items()})
        print("\nClassification report:\n", report)
        print("\nConfusion matrix:\n", cm)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            f.write("Metrics\n")
            f.write("=======\n")
            for k, v in metrics.items():
                f.write(f"{k}: {v:.6f}\n")
            f.write("\n\nClassification Report\n")
            f.write("=====================\n")
            f.write(report)
            f.write("\n\nConfusion Matrix\n")
            f.write("================\n")
            f.write(np.array2string(cm))

    return metrics, report, cm


def run_epoch(model, loader, criterion=None, optimizer=None, train=True):
    model.train(train)
    running_loss, running_correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        with torch.set_grad_enabled(train):
            logits = model(imgs)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        total += imgs.size(0)

    return running_loss / total, running_correct / total
