"""Process: MediaPipe keypoint extraction -> raw jazz landmarks streamed to STDOUT.

Runs the (GPU) MediaPipe extraction and writes an in-memory .npz of the RAW jazz
landmarks (world `metricScaleNodes` + image `nodes`, untouched) to STDOUT as raw
bytes. All human/progress output goes to STDERR so STDOUT stays a clean binary
stream -- the trainer reads it from a pipe (see run.py) and calls load_payload(),
which is where config.FEATURES turns landmarks into a feature vector.

Keeping the payload raw is deliberate: ONE cache then serves every FEATURES mode,
so a 2d run and a 2.5d+3d run are scored on byte-identical underlying data.

    python extract.py --sources 6 | python train.py
    python extract.py --dir test_set > test_set.npz
    python extract.py --selfcheck        # payload round-trip, no MediaPipe needed
"""

import argparse
import io
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "neural_network"))

import config as cfg                                      # noqa: E402
from config import KP2D_COLS, KP3D_COLS, DEFAULT_LIMIT, IMAGE_SOURCES  # noqa: E402

# npz keys of the STDOUT/cache payload, in the order _hands() returns them.
PAYLOAD_KEYS = ("world", "image", "handedness", "y", "paths")


def hand_feature(world_xyz, image_xyz, handedness=""):
    """One hand's feature vector, honoring config.FEATURES / config.FLIP_LEFT_TO_RIGHT.

    world_xyz: (21,3) world landmarks; image_xyz: (21,3) image x,y in [0,1] + depth z.
    Left hands are mirrored to right chirality (world x negated, image x -> 1-x; z is
    depth, unaffected by a horizontal mirror).
    Shared by build_features (offline) and app.py (live) so both always match.
    """
    w = np.asarray(world_xyz, np.float32).copy()
    xy = np.asarray(image_xyz, np.float32).copy()
    if cfg.FLIP_LEFT_TO_RIGHT and str(handedness).lower().startswith("l"):
        w[:, 0] = -w[:, 0]
        xy[:, 0] = 1.0 - xy[:, 0]
    feats = cfg.FEATURES
    parts = []
    if "3d" in feats:
        mn, mx = w.min(), w.max()                             # global min/max over all 63 values (not per-axis)
        w3 = (w - mn) / (mx - mn) if mx > mn else np.zeros_like(w)  # min-max -> [0,1], matches 2D range
        parts.append(w3.reshape(-1))                          # 63 min-max normalized world landmarks
    if "2.5d" in feats:
        parts.append(xy.reshape(-1))                          # 63 image x,y + relative depth z
    elif "2d" in feats:
        parts.append(xy[:, :2].reshape(-1))                   # 42 image x,y (no depth)
    if not parts:
        raise ValueError(f"config.FEATURES must be one of 2d/2.5d/3d/2d+3d/2.5d+3d (got {feats!r})")
    return np.concatenate(parts).astype(np.float32)


def _hands(df: pd.DataFrame):
    """Rows with a detected hand -> (world, image, handedness, label, path), raw.

    Order matches PAYLOAD_KEYS. `world`/`image` are (n,21,3) exactly as MediaPipe
    (and so jazz) produced them -- no flip, no normalization.
    """
    d = df[(df.detected == 1) & df[KP3D_COLS].notna().all(axis=1)].reset_index(drop=True)
    return (d[KP3D_COLS].to_numpy(np.float32).reshape(-1, 21, 3),   # jazz metricScaleNodes
            d[KP2D_COLS].to_numpy(np.float32).reshape(-1, 21, 3),   # jazz nodes (+ depth z)
            d.handedness.to_numpy().astype("U"),
            d.label.to_numpy().astype("U"),
            d.path.to_numpy().astype("U"))


def features_from(world, image, handedness):
    """(n,21,3)+(n,21,3) raw landmarks -> (n, dim) features per config.FEATURES."""
    zero = np.zeros((cfg.NUM_KP, 3), np.float32)
    if len(world) == 0:                              # empty set (e.g. nothing detected)
        return np.zeros((0, hand_feature(zero, zero).size), np.float32)
    return np.stack([hand_feature(w, n, h)
                     for w, n, h in zip(world, image, handedness)]).astype(np.float32)


def build_features(df: pd.DataFrame, with_paths: bool = False):
    """Per detected hand: feature vector per config.FEATURES (42/63/63/105/126 dims).

    with_paths=True also returns the source image path per row (aligned with X),
    which image+keypoint models need to pair pixels with the detected keypoints.
    """
    world, image, hnd, y, paths = _hands(df)
    X = features_from(world, image, hnd)
    return (X, y, paths) if with_paths else (X, y)


def dump_payload(df: pd.DataFrame) -> bytes:
    """The STDOUT/cache payload for `df`: an in-memory .npz of raw landmarks."""
    buf = io.BytesIO()
    np.savez(buf, **dict(zip(PAYLOAD_KEYS, _hands(df))))
    return buf.getvalue()


def load_payload(raw: bytes):
    """(X, y, paths) from an extract.py payload, featurized per config.FEATURES.

    Every consumer goes through here, so switching FEATURES needs no re-extraction
    and cannot silently pair one mode's features with another's cache.
    """
    d = np.load(io.BytesIO(raw))
    return (features_from(d["world"], d["image"], d["handedness"]),
            d["y"].astype(str), d["paths"].astype(str))


_FEATURE_DIMS = {"2d": 42, "2.5d": 63, "3d": 63, "2d+3d": 105, "2.5d+3d": 126}


def selfcheck() -> None:
    """The payload must round-trip to the same features a live DataFrame gives, for
    EVERY mode -- that is the whole reason one cache can serve them all."""
    rng = np.random.default_rng(0)
    rows = [["A" if i % 2 else "B", "2", "data", 1, "Left" if i % 3 else "Right", 0.9,
             f"/img{i}.jpg", *rng.random(63).astype(np.float32),
             *rng.normal(0, 0.05, 63).astype(np.float32)] for i in range(40)]
    rows.append(["C", "2", "data", 0, "", np.nan, "", *([np.nan] * 126)])   # undetected
    df = pd.DataFrame(rows, columns=cfg.ALL_COLS)

    raw = dump_payload(df)
    assert set(np.load(io.BytesIO(raw)).files) == set(PAYLOAD_KEYS)
    for mode, dim in _FEATURE_DIMS.items():
        cfg.FEATURES = mode
        X, y, _ = load_payload(raw)
        assert X.shape == (40, dim), f"{mode}: {X.shape} != {(40, dim)}"
        assert (y != "C").all(), f"{mode}: undetected rows must be dropped"
        Xlive, ylive = build_features(df)
        assert np.array_equal(X, Xlive) and np.array_equal(y, ylive), f"{mode}: cache != live"
        assert dump_payload(df) == raw, "payload must not depend on config.FEATURES"
        print(f"  {mode:<9}{dim:>4} dims   cache == live extraction")
    assert load_payload(dump_payload(df.tail(1)))[0].shape == (0, dim), "empty set"
    print("selfcheck OK")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selfcheck", action="store_true",
                    help="verify the payload round-trip (offline, no images)")
    ap.add_argument("--sources", nargs="+", default=IMAGE_SOURCES,
                    help="dataset sources to extract (default: config.IMAGE_SOURCES)")
    ap.add_argument("--dir", default=None,
                    help="extract an arbitrary <CLASS>/<images> tree instead of --sources")
    ap.add_argument("--limit", type=int, default=None,
                    help="max images per source, per class if config.SAMPLE_PER_CLASS "
                         "(0 = all). Default: config.DEFAULT_LIMIT, or EVAL_LIMIT with --dir")
    a = ap.parse_args()
    if a.selfcheck:
        selfcheck()
        return
    # an eval tree is scored whole by default; a training source is sampled
    limit = a.limit if a.limit is not None else (cfg.EVAL_LIMIT if a.dir else DEFAULT_LIMIT)

    # keep STDOUT clean for the binary payload: route all status/progress to STDERR
    payload_out = sys.stdout
    sys.stdout = sys.stderr
    from dataloader import extract_dir, extract_frame   # imports MediaPipe; keep it out
    #                                                    of importers that only decode
    df = (extract_dir(a.dir, limit=limit) if a.dir
          else extract_frame(a.sources, limit=limit))
    payload = dump_payload(df)
    n_det = int((df.detected == 1).sum())
    print(f"extracted {n_det} hands from {len(df)} images "
          f"({df[df.detected == 1].label.nunique()} classes)")
    sys.stdout = payload_out

    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
