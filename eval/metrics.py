import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


def per_position_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, list[float]]:
    """
    Compute accuracy, precision, recall, and MCC per digit position.

    y_true, y_pred: shape (n_samples, 4)
    """
    metrics = {"accuracy": [], "precision": [], "recall": [], "mcc": []}
    for pos in range(4):
        true_col = y_true[:, pos]
        pred_col = y_pred[:, pos]
        metrics["accuracy"].append(float(accuracy_score(true_col, pred_col)))
        metrics["precision"].append(
            float(
                precision_score(
                    true_col, pred_col, average="macro", zero_division=0
                )
            )
        )
        metrics["recall"].append(
            float(
                recall_score(
                    true_col, pred_col, average="macro", zero_division=0
                )
            )
        )
        metrics["mcc"].append(float(matthews_corrcoef(true_col, pred_col)))
    return metrics


def exact_match_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).all(axis=1).mean())


def confusion_matrices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> list[np.ndarray]:
    return [
        confusion_matrix(y_true[:, pos], y_pred[:, pos], labels=list(range(10)))
        for pos in range(4)
    ]
