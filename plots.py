"""Shared PNG plot helpers: training-history curves and confusion matrices.

Headless (Agg backend) so it works over SSH and inside subprocesses. Trainers
call save_history() after fit; test.py calls save_confusion() per test set. Every
model writes its images beside its weights, named from the weights stem.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")                     # no display -> must be set before pyplot
import matplotlib.pyplot as plt           # noqa: E402


def save_history(history, out_dir, stem):
    """From a fit-history dict, save <stem>_accuracy.png and <stem>_loss.png.

    history: dict of lists under 'accuracy'/'val_accuracy'/'loss'/'val_loss'
    (the val_* keys are optional). Returns the paths written.
    """
    n = len(history.get("loss") or history.get("accuracy") or [])
    epochs = range(1, n + 1)
    written = []
    for metric, title in (("accuracy", "Accuracy"), ("loss", "Loss")):
        if not history.get(metric):
            continue
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(epochs, history[metric], marker=".", label=f"train {metric}")
        if history.get(f"val_{metric}"):
            ax.plot(epochs, history[f"val_{metric}"], marker=".", label=f"val {metric}")
        ax.set_xlabel("epoch")
        ax.set_ylabel(metric)
        ax.set_title(f"{stem} \u2014 {title}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        dest = os.path.join(out_dir, f"{stem}_{metric}.png")
        fig.tight_layout()
        fig.savefig(dest, dpi=120)
        plt.close(fig)
        written.append(dest)
    return written


def save_val_predictions(y_true, y_pred, out_dir, stem):
    """Dump the held-out validation predictions (string labels) to <stem>_val.npz.

    Trainers call this after fit so test.py can report the REAL validation split
    (not a re-extracted sample that may overlap training). Returns the path.
    """
    import numpy as np
    dest = os.path.join(out_dir, f"{stem}_val.npz")
    np.savez(dest, y_true=np.asarray(y_true).astype("U"),
             y_pred=np.asarray(y_pred).astype("U"))
    return dest


def save_confusion(y_true, y_pred, out_path, title, labels=None):
    """Save a confusion-matrix PNG from string label arrays. Returns out_path."""
    import numpy as np
    from sklearn.metrics import ConfusionMatrixDisplay

    if len(y_true) == 0:
        return None
    if labels is None:
        labels = sorted(set(np.asarray(y_true).tolist()) | set(np.asarray(y_pred).tolist()))
    side = max(6.0, len(labels) * 0.5)
    fig, ax = plt.subplots(figsize=(side, side))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=labels, ax=ax, colorbar=False,
        xticks_rotation="vertical", values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
