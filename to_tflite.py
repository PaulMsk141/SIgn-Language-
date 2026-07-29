"""Export a trained MLP (.joblib from train.py) to .tflite, beside the joblib.

    python to_tflite.py                              # config.MODEL_WEIGHTS
    python to_tflite.py neural_network/asl_mlp_3d.joblib

The .tflite eats the very same feature vector app.py builds (config.FEATURES, so 63
floats for 3d) and returns softmax probabilities: standardisation, the ReLU stack and
the softmax are all baked into the graph, so a caller needs no numpy-side pre/post
step. Only the class names can't live in a tflite tensor, so they go next to it in
<stem>_labels.txt, one per line, in output-column order.

Conversion is exact (float32 constants, no quantisation) and the script asserts it by
comparing against the joblib before writing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent / "neural_network"))  # unpickle ASLClassifier


def to_tflite(clf) -> bytes:
    W = [tf.constant(w) for w in clf.weights]
    B = [tf.constant(b) for b in clf.biases]

    # Batch 1: the app classifies one hand per frame, and a fixed shape keeps every
    # runtime happy (no ResizeInputTensor call before the first Invoke).
    @tf.function(input_signature=[tf.TensorSpec([1, len(clf.mean)], tf.float32, "features")])
    def forward(x):
        x = (x - clf.mean) / clf.std
        for w, b in zip(W[:-1], B[:-1]):
            x = tf.nn.relu(tf.matmul(x, w) + b)
        return tf.nn.softmax(tf.matmul(x, W[-1]) + B[-1])

    return tf.lite.TFLiteConverter.from_concrete_functions(
        [forward.get_concrete_function()]).convert()


def check(clf, blob, n=256, tol=1e-5) -> float:
    """Worst |tflite - joblib| probability over n random features; raises if > tol."""
    # Features are min-max normalised per frame, so U[0,1] covers the real input range.
    X = np.random.default_rng(0).random((n, len(clf.mean)), dtype=np.float32)
    interp = tf.lite.Interpreter(model_content=blob)
    interp.allocate_tensors()
    i, o = interp.get_input_details()[0]["index"], interp.get_output_details()[0]["index"]

    got = np.empty((n, len(clf.classes_)), np.float32)
    for k, row in enumerate(X):
        interp.set_tensor(i, row[None])
        interp.invoke()
        got[k] = interp.get_tensor(o)

    want = clf.predict_proba(X)
    worst = float(np.abs(got - want).max())
    assert worst < tol, f"tflite disagrees with the joblib by {worst}"
    assert (clf.classes_[got.argmax(1)] == clf.predict(X)).all()
    return worst


def main() -> None:
    if len(sys.argv) > 1:
        src = Path(sys.argv[1])
    else:
        import config as cfg
        src = Path(cfg.MODEL_WEIGHTS)

    clf = joblib.load(src)
    blob = to_tflite(clf)
    print(f"max probability error vs {src.name}: {check(clf, blob):.2e}")

    dest = src.with_suffix(".tflite")
    dest.write_bytes(blob)
    labels = src.with_name(f"{src.stem}_labels.txt")
    labels.write_text("\n".join(clf.classes_) + "\n")
    print(f"saved -> {dest} ({dest.stat().st_size / 1024:.0f} KB), "
          f"{len(clf.classes_)} labels -> {labels}")


if __name__ == "__main__":
    main()
