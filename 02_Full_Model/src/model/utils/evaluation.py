"""
Evaluation utilities for classification models.

This module provides a function `evaluate` to compute and save classification
metrics (accuracy, AUC, confusion matrix, classification report), as well as
visualizations (ROC curve and confusion matrix heatmap). It saves the results
as `.txt` and `.png` files under a given output directory.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize


def evaluate(y_pred, y_proba, X, y, tag, run_dir: Path):
    """
    Evaluate classification performance, plot ROC and confusion matrix,
    and save all results under `run_dir`.

    Args:
        y_pred (array-like): Predicted class labels.
        y_proba (array-like): Predicted class probabilities (n_samples, n_classes).
        X (pd.DataFrame): Feature matrix (used for shape only).
        y (array-like): True class labels.
        tag (str): Either 'Train' or 'Test', used to label output files.
        run_dir (Path): Directory to save output plots and metrics.
    """
    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    print(f"{tag} Accuracy: {acc:.4f}\n")

    aucs = []
    if y_proba is not None:
        classes = np.unique(y)
        y_bin = label_binarize(y, classes=classes)
        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            aucs.append(auc(fpr, tpr))
            plt.plot(fpr, tpr, label=f"Class {cls} AUC={aucs[-1]:.2f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.title(f"{tag} ROC")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend()
        plt.tight_layout()
        plt.savefig(run_dir / f"roc_{tag}.png")
        plt.close()

        print(f"{tag} Average ROC AUC: {np.mean(aucs):.4f}")
    else:
        print("No probability scores provided – skipping AUC.")

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{tag} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(run_dir / f"cm_{tag}.png")
    plt.close()

    metrics_path = run_dir / f"metrics_{tag}.txt"
    with open(metrics_path, "w") as f:
        f.write(f"{tag} Classification Report\n")
        f.write("=" * 40 + "\n")
        f.write(classification_report(y, y_pred))
        f.write("\n\nAccuracy: {:.4f}".format(acc))
        if aucs:
            f.write("\nAverage ROC AUC: {:.4f}".format(np.mean(aucs)))
        f.write("\n\nConfusion Matrix:\n")
        for row in cm:
            f.write(" ".join(map(str, row)) + "\n")
