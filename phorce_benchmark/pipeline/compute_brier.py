"""
compute_brier.py — correct, bounded Brier score for the PHORCE models.

The Brier score is the mean squared error between predicted PROBABILITIES and the
0/1 outcome. It is bounded:
    * binary                     : [0, 1]
    * multiclass (sum-of-squares): [0, 2]
Any reported "Brier" > 1 for a binary model is a bug. The usual causes:
    * feeding decision_function margins or raw scores instead of predict_proba()
    * feeding predicted class labels instead of probabilities
    * summing squared errors instead of averaging (missing 1/N)
    * reporting log_loss (which is unbounded above) under the "Brier" label

Use `predict_proba(X)[:, 1]` for the positive-class probability in binary models.
"""
import numpy as np
from sklearn.metrics import brier_score_loss


def brier_binary(y_true, proba_pos):
    """Binary Brier score. proba_pos = P(class=1), e.g. clf.predict_proba(X)[:, 1]."""
    y_true = np.asarray(y_true).astype(int)
    proba_pos = np.asarray(proba_pos, dtype=float)
    if proba_pos.min() < 0 or proba_pos.max() > 1:
        raise ValueError(
            "proba_pos must be probabilities in [0,1]; got range "
            f"[{proba_pos.min():.3g}, {proba_pos.max():.3g}]. "
            "Pass predict_proba()[:,1], not decision_function()/labels."
        )
    return float(brier_score_loss(y_true, proba_pos))


def brier_multiclass(y_true, proba):
    """Generalized (sum-of-squares) multiclass Brier, bounded [0,2].

    proba : array (n_samples, n_classes) from clf.predict_proba(X).
    """
    proba = np.asarray(proba, dtype=float)
    n, k = proba.shape
    onehot = np.zeros((n, k))
    onehot[np.arange(n), np.asarray(y_true).astype(int)] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def brier_baserate(y_true):
    """Brier of a naive constant predictor that always outputs the base rate.
    Useful as the reference a real model must beat."""
    y_true = np.asarray(y_true).astype(int)
    p = y_true.mean()
    return float(brier_score_loss(y_true, np.full_like(y_true, p, dtype=float)))
