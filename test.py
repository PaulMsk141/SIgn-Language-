"""Offline test: evaluate the config.MODEL model on TWO sets, per-class metrics.

Own process. Handles every model kind uniformly:
  * mlp             -> joblib ASLClassifier (numpy), keypoint features
  * mini_cnn        -> .keras model, keypoint features (MediaPipe builds them here)
  * cnn/inceptionv3 -> .keras model, raw images (no MediaPipe)
  * mini_cnn+cnn    -> .keras model, images + keypoints (MediaPipe builds them here)

A keypoint .keras model runs TF on CPU here, because this same process also runs
MediaPipe for extraction and keeping TF off CUDA avoids a clash. The image model
runs no MediaPipe, so it uses the A100. Two sets are scored:

  1. validation: the trainer's REAL held-out split (loaded from <stem>_val.npz), so
     it never reuses training images (no re-extraction / memorization). Training
     diagnostic only.
  2. config.TEST_SET_DIR -- test_set/<left|right>/<CLASS>/<frames>. This is the ONE
     held-out set every reported score comes from; frames with no class folder are
     unlabeled and are ignored.

Labels compare case-insensitively. Each section reports per-class precision, recall,
and f1-score plus a final accuracy (no support/macro/weighted rows). The report prints
and saves to the model's own folder (config.MODEL_DIR/<weights>_test.txt).
Weights = config.MODEL_WEIGHTS.

    python test.py --sources 6
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "neural_network"))

from sklearn.metrics import accuracy_score, classification_report  # noqa: E402

import config as cfg                                     # noqa: E402
from extract import build_features                       # noqa: E402
from dataloader import extract_dir, _images, _in_scope, _sample   # noqa: E402


def load_model():
    """(kind, model, classes). kind in {'joblib','keras'}; classes=None for joblib."""
    w = str(cfg.MODEL_WEIGHTS)
    if w.endswith(".joblib"):
        import joblib
        return "joblib", joblib.load(w), None
    if "keypoints" in cfg.MODELS[cfg.MODEL]["input"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""          # shares process w/ MediaPipe -> TF off CUDA
    else:
        from gpu_env import enable_tf_gpu
        enable_tf_gpu()                                  # image model: no MediaPipe -> use the A100
    import tensorflow as tf
    model = tf.keras.models.load_model(w)
    classes = json.loads(Path(os.path.splitext(w)[0] + "_classes.json").read_text())
    return "keras", model, classes


def image_dir_items(root, limit):
    """(paths, labels) under root/<CLASS>/<img>, same scope/sampling as extract_dir."""
    items = _sample([(str(p), p.parent.name, "eval") for p in _images(Path(root))
                     if _in_scope(p.parent.name)], limit)
    up = cfg.UPPERCASE_LABELS
    return [it[0] for it in items], [it[1].upper() if up else it[1] for it in items]


def score_iterations(X, y, stem) -> None:
    """Score every training iteration on test_set, into the trainer's <stem>_runs.joblib.

    The trainer knows the ITERATIONS models but not test_set; this process has just
    extracted test_set anyway, so it adds the predictions here. That file is the whole
    input to experiments.ipynb, which therefore reports without retraining anything.
    """
    p = Path(cfg.MODEL_DIR) / f"{stem}_runs.joblib"
    if not p.is_file() or len(y) == 0:
        return
    import joblib
    d = joblib.load(p)
    if "models" not in d:                                 # already scored
        return
    d["y_true"] = np.char.upper(np.asarray(y).astype("U"))
    # pop: the scored file then holds arrays only, so the notebook can load it without
    # ASLClassifier (or torch, or this repo) importable at all.
    d["y_pred"] = [np.char.upper(m.predict(X).astype("U")) for m in d.pop("models")]
    d["accuracy"] = [float(accuracy_score(d["y_true"], yp)) for yp in d["y_pred"]]
    joblib.dump(d, p)
    print(f"{len(d['accuracy'])} iterations scored on test_set "
          f"(mean accuracy {np.mean(d['accuracy']):.4f}) -> {p}")


def report(name, y_true, y_pred) -> str:
    """Per-class precision/recall/f1 + final accuracy only (no support/macro/weighted)."""
    if len(y_true) == 0:
        return f"\n=== {name}: no samples to score ==="
    yt = np.char.upper(np.asarray(y_true).astype("U"))
    yp = np.char.upper(np.asarray(y_pred).astype("U"))
    d = classification_report(yt, yp, zero_division=0, digits=3, output_dict=True)
    acc = d.get("accuracy", accuracy_score(yt, yp))
    labels = sorted(k for k in d if k not in ("accuracy", "macro avg", "weighted avg"))
    lines = [f"\n=== {name}: {len(yt)} samples ===",
             f"{'class':<8}{'precision':>11}{'recall':>11}{'f1-score':>11}"]
    for lab in labels:
        m = d[lab]
        lines.append(f"{lab:<8}{m['precision']:>11.3f}{m['recall']:>11.3f}{m['f1-score']:>11.3f}")
    lines.append(f"\naccuracy: {acc:.4f}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sources", nargs="+", default=cfg.IMAGE_SOURCES)
    ap.add_argument("--limit", type=int, default=cfg.TEST_LIMIT,
                    help="images per source for the sampled test (0 = all)")
    ap.add_argument("--seed", type=int, default=cfg.TEST_SEED,
                    help="sampling seed for the sampled test (fresh vs training)")
    ap.add_argument("--eval-limit", type=int, default=cfg.EVAL_LIMIT,
                    help="images per class from test_set/ (0 = all)")
    a, _unknown = ap.parse_known_args()                   # tolerate legacy --sources/--limit

    kind, model, classes = load_model()
    input_type = cfg.MODELS[cfg.MODEL]["input"]
    is_image = input_type == "image"
    is_concat = input_type == "image+keypoints"           # needs both pixels and keypoints
    weights = str(cfg.MODEL_WEIGHTS)
    stem = os.path.splitext(os.path.basename(weights))[0]

    def predict_kp(X):
        if kind == "joblib":
            return model.predict(X)
        return np.array(classes)[model.predict(X, verbose=0).argmax(1)]

    def predict_images(paths):
        import tensorflow as tf
        size = model.input_shape[1]                       # the model's own H (e.g. 299)

        def load(p):
            img = tf.io.decode_image(tf.io.read_file(p), channels=3, expand_animations=False)
            return tf.cast(tf.image.resize(img, (size, size)), tf.float32)

        ds = (tf.data.Dataset.from_tensor_slices(paths)
              .map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(cfg.BATCH_SIZE))
        return np.array(classes)[model.predict(ds, verbose=0).argmax(1)]

    def predict_concat(X, paths):
        import tensorflow as tf
        img_shape = next(s for s in model.input_shape if len(s) == 4)  # image branch (H, ch)
        size, ch = img_shape[1], img_shape[3]

        def load(p):
            img = tf.io.decode_image(tf.io.read_file(p), channels=3, expand_animations=False)
            img = tf.image.resize(img, (size, size))
            if ch == 1:                                   # grayscale image branch
                img = tf.image.rgb_to_grayscale(img)
            return tf.cast(img, tf.float32)

        imgs = np.stack(list(tf.data.Dataset.from_tensor_slices(paths)
                             .map(load, num_parallel_calls=tf.data.AUTOTUNE).as_numpy_iterator()))
        # feed both inputs explicitly (a bare (img, kp) dataset gets misread as (x, y))
        inp = [imgs if len(t.shape) == 4 else X for t in model.inputs]
        return np.array(classes)[model.predict(inp, batch_size=cfg.BATCH_SIZE, verbose=0).argmax(1)]

    scored = []                                           # test_set features, reused below

    def eval_set(name, tag, df_getter, img_getter) -> str:
        if is_image:
            paths, y = img_getter()
            yp = predict_images(paths) if paths else []
        elif is_concat:
            X, y, paths = build_features(df_getter(), with_paths=True)
            keep = paths != ""                            # need real image files for the cnn branch
            X, y, paths = X[keep], y[keep], paths[keep]
            yp = predict_concat(X, list(paths)) if len(y) else []
        else:
            X, y = build_features(df_getter())
            yp = predict_kp(X) if len(y) else []
            scored.append((X, y))
        if len(y):                                        # save a confusion-matrix PNG
            from plots import save_confusion
            cm = os.path.join(cfg.MODEL_DIR, f"{stem}_confusion_{tag}.png")
            save_confusion(np.char.upper(np.asarray(y).astype("U")),
                           np.char.upper(np.asarray(yp).astype("U")),
                           cm, f"{cfg.MODEL} \u2014 {name}")
            print(f"confusion matrix -> {cm}")
        return report(name, y, yp)

    parts = [f"# test report for {weights}  ({datetime.now():%Y-%m-%d %H:%M:%S})"]

    # upper: the REAL held-out validation split (dumped by the trainer), never a
    # re-extracted sample that could reuse training images.
    valp = Path(cfg.MODEL_DIR) / f"{stem}_val.npz"
    if valp.is_file():
        d = np.load(valp)
        yt, yp = d["y_true"], d["y_pred"]
        from plots import save_confusion
        cm = os.path.join(cfg.MODEL_DIR, f"{stem}_confusion_val.png")
        save_confusion(np.char.upper(yt.astype("U")), np.char.upper(yp.astype("U")),
                       cm, f"{cfg.MODEL} \u2014 validation")
        print(f"confusion matrix -> {cm}")
        parts.append(report("validation", yt, yp))
    else:
        parts.append(f"\n(no validation sidecar {valp.name}; retrain to generate it)")

    # the held-out set: test_set/<left|right>/<CLASS>/<frames>. rglob + parent-folder
    # label means unlabeled frames (parent left/right) drop out.
    if cfg.TEST_SET_DIR.is_dir():
        parts.append(eval_set(
            "test_set", "testset",
            lambda: extract_dir(cfg.TEST_SET_DIR, limit=a.eval_limit),
            lambda: image_dir_items(cfg.TEST_SET_DIR, a.eval_limit)))
    else:
        parts.append(f"\n(skipping test_set: {cfg.TEST_SET_DIR} not found)")

    if scored:
        score_iterations(*scored[-1], stem)

    text = "\n".join(parts)
    print(text)
    dest = os.path.join(cfg.MODEL_DIR, f"{stem}_test.txt")
    Path(dest).write_text(text + "\n")
    print(f"\nsaved report -> {dest}")


if __name__ == "__main__":
    main()
