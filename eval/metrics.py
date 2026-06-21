import numpy as np
from sklearn.metrics import confusion_matrix


def exact_match_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).all(axis=1).mean())


def confusion_matrix_aggregate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    true_flat = y_true.reshape(-1)
    pred_flat = y_pred.reshape(-1)
    return confusion_matrix(true_flat, pred_flat, labels=list(range(10)))
