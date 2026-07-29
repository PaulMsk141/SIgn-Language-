"""Inference wrapper for the GPU-trained ASL MLP -- pure numpy, no torch needed.

train.py trains the net with PyTorch on the GPU, then exports its weights into
this small class. It quacks like the old sklearn classifier (`predict`,
`predict_proba`, `classes_`), so app.py loads it unchanged:

    python app.py --weights asl_mlp.joblib

Input is the same feature app.py builds (config.FEATURES):
e.g. "2.5d+3d" = min-max world landmarks (63) + image xyz with depth (63) = 126.
"""

from __future__ import annotations

import numpy as np


class ASLClassifier:
    """A plain MLP (ReLU hidden layers, softmax head) evaluated in numpy."""

    def __init__(self, weights, biases, classes, mean, std):
        self.weights = [np.asarray(w, np.float32) for w in weights]
        self.biases = [np.asarray(b, np.float32) for b in biases]
        self.classes_ = np.asarray(classes)
        self.mean = np.asarray(mean, np.float32)
        self.std = np.asarray(std, np.float32)

    def _logits(self, X):
        X = (np.asarray(X, np.float32) - self.mean) / self.std
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            X = np.maximum(0.0, X @ W + b)          # ReLU
        return X @ self.weights[-1] + self.biases[-1]

    def predict_proba(self, X):
        z = self._logits(X)
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self._logits(X).argmax(axis=1)]
