import json
import os
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)


class ClassificationEvaluator:
    def __init__(self, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

    @torch.no_grad()
    def evaluate(
        self,
        model: Optional[torch.nn.Module] = None,
        loader: Optional[Iterable] = None,
        *,
        averages: List[str] = ("macro", "micro"),
        results_path: Optional[str] = "./",
        save_confusion: bool = False,
        y_true=None,
        y_pred=None,
        y_prob=None,
    ) -> Tuple[
        Dict[str, Any], np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]
    ]:
        """
        Two usage modes:

        1) Inference mode:
            evaluate(model=..., loader=...)
           -> computes y_true/y_pred/y_prob and metrics

        2) Precomputed mode:
            evaluate(y_true=..., y_prob=...) or evaluate(y_true=..., y_pred=..., y_prob=...)
           -> computes missing pieces and metrics

        Returns:
            metrics, confusion_matrix, y_true, y_pred, y_prob
        """

        # --------------------
        # Decide how to obtain y_true/y_pred/y_prob
        # --------------------
        os.makedirs(Path(results_path), exist_ok=True)
        if model is not None:
            if loader is None:
                raise ValueError("If model is provided, loader must also be provided.")
            logits, targets = self._infer(loader, model, results_path)

            y_true = targets
            y_prob = self._softmax(logits)  # shape [N, C]
            y_pred = logits.argmax(axis=1)

        else:
            # Precomputed mode: must at least have y_true and (y_pred or y_prob)
            if y_true is None:
                raise ValueError("Provide y_true when model is None.")
            if y_pred is None and y_prob is None:
                raise ValueError(
                    "Provide at least one of y_pred or y_prob when model is None."
                )

            y_true = np.asarray(y_true)

            if y_prob is not None:
                y_prob = np.asarray(y_prob)
                if y_prob.ndim != 2:
                    raise ValueError("y_prob must be a 2D array of shape [N, C].")

            if y_pred is None and y_prob is not None:
                y_pred = y_prob.argmax(axis=1)
            elif y_pred is not None:
                y_pred = np.asarray(y_pred)

        # Basic checks
        if len(y_true) != len(y_pred):
            raise ValueError(
                f"Length mismatch: y_true({len(y_true)}) vs y_pred({len(y_pred)})"
            )
        if y_prob is not None and len(y_true) != y_prob.shape[0]:
            raise ValueError(
                f"Length mismatch: y_true({len(y_true)}) vs y_prob({y_prob.shape[0]})"
            )

        # Determine number of classes
        if y_prob is not None:
            num_classes = y_prob.shape[1]
        else:
            # fallback: infer from labels/preds
            num_classes = int(max(np.max(y_true), np.max(y_pred))) + 1

        metrics: Dict[str, Any] = {}

        # --------------------
        # Accuracy
        # --------------------
        metrics["accuracy"] = accuracy_score(y_true, y_pred)

        # --------------------
        # Precision / Recall / F1 / AUC (OvR)
        # --------------------
        for avg in averages:
            p, r, f1, _ = precision_recall_fscore_support(
                y_true,
                y_pred,
                average=avg,
                zero_division=0,
            )
            metrics[f"precision_{avg}"] = float(p)
            metrics[f"recall_{avg}"] = float(r)
            metrics[f"f1_{avg}"] = float(f1)

            # AUC needs probabilities
            if y_prob is None:
                metrics[f"auc_{avg}_ovr"] = float("nan")
                continue

            # one-hot for roc_auc_score
            y_true_oh = np.eye(num_classes)[y_true]

            try:
                metrics[f"auc_{avg}_ovr"] = float(
                    roc_auc_score(
                        y_true_oh,
                        y_prob,
                        average=avg,
                        multi_class="ovr",
                    )
                )
            except Exception:
                metrics[f"auc_{avg}_ovr"] = float("nan")

        # --------------------
        # MCC
        # --------------------
        metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred))

        # --------------------
        # Confusion matrix
        # --------------------
        cm = confusion_matrix(y_true, y_pred)

        if save_confusion and results_path is not None:
            np.save(self._check_path(results_path) / "confusion.npy", cm)

        if results_path is not None:
            payload = {"metrics": metrics, "confusion_matrix": cm.tolist()}
            with open(Path(results_path) / "metrics.json", "w") as f:
                json.dump(payload, f, indent=2)

        return metrics, cm

    # --------------------------------------------------
    # Inference
    # --------------------------------------------------
    def _infer(self, loader, model, results_path):
        results_path = Path(results_path)
        logits_path = results_path / "logits.npy"
        targets_path = results_path / "targets.npy"

        model.eval()
        model.to(self.device)

        all_logits, all_targets = [], []

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                logits = model(images)
                all_logits.append(logits.detach().cpu().numpy())
                all_targets.append(labels.detach().cpu().numpy())

        logits = np.concatenate(all_logits, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        # Save for convenience (optional)
        np.save(logits_path, logits)
        np.save(targets_path, targets)

        return logits, targets

    # --------------------------------------------------
    # Utils
    # --------------------------------------------------
    @staticmethod
    def _softmax(x: np.ndarray):
        x = x - x.max(axis=1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=1, keepdims=True)

    @staticmethod
    def _check_path(path):
        path = Path(path)
        os.makedirs(path, exist_ok=True)
        return path
