"""Extract MediaPipe hand keypoints (jazz format) into an in-memory DataFrame.

Each row = one image (or one folder-14 CSV row):

    kp2d  21x3 normalized image landmarks (cols n0..n62)   -> jazz `nodes`
    kp3d  21x3 metric world landmarks     (cols w0..w62)   -> jazz `metricScaleNodes`
    handedness  Left/Right/""    score  hand confidence    label / source / split

Nothing is written to disk. Build the DataFrame incrementally, one source at a
time, so peak RAM stays bounded:

    from dataloader import extract_frame
    df = extract_frame("2")           # extract source 2
    df = extract_frame("3", df)       # append source 3 onto it
    df = extract_frame("2", df)       # re-run -> source-2 rows are replaced

Sources: "2".."13" are images (MediaPipe runs now), "14" is a landmark CSV
(kp2d only, kp3d = NaN). Folder 1 is excluded. All knobs live in config.py.

Speed / RAM: single process on the GPU delegate (workers=0) is ~200 img/s and
uses ~0.9 GB. Each CPU worker also loads its own MediaPipe+TF (~0.9 GB), so
`workers=N` needs ~N*0.9 GB of RAM. Measure both with benchmark_workers().

    python dataloader.py --sources 2 14        # extract, print summary
    python dataloader.py --benchmark           # compare worker counts on a sample
    python dataloader.py --selftest            # tiny sanity check
"""

from __future__ import annotations

import argparse
import random
import re
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config as cfg


# --------------------------------------------------------------------------- #
# Source enumeration: (image_path, label, split) per source.                  #
# --------------------------------------------------------------------------- #
def _images(root: Path):
    return (
        p for p in root.rglob("*")
        if p.suffix.lower() in cfg.IMAGE_EXTS
        and not p.name.startswith("._")
        and "__MACOSX" not in p.parts
    )


def _split_of(path: Path) -> str:
    low = [p.lower() for p in path.parts]
    if any("test" in p for p in low):
        return "test"
    if any("train" in p for p in low):
        return "train"
    if any("valid" in p for p in low):
        return "val"
    return "data"


def _heuristic_label(path: Path) -> str:
    label = path.parent.name
    if label.lower() in cfg.SPLIT_CONTAINERS:
        token = path.stem.split("_")[0]
        m = re.match(r"[A-Za-z]+", token)
        label = m.group(0) if m else token
    return label


def iter_source(source: str):
    if source == "12":
        root = cfg.DATASETS / cfg.SOURCE_ROOTS["12"]
        for p in _images(root):
            yield p, p.relative_to(root).parts[0], "data"
    elif source == "13":
        root = cfg.DATASETS / cfg.SOURCE_ROOTS["13"]
        for cls in sorted(d for d in root.iterdir() if d.is_dir()):
            sub = cls / cfg.FOLDER13_STYLE
            if sub.is_dir():
                for p in _images(sub):
                    yield p, cls.name, "data"
    elif source in cfg.VIDEO_SOURCES:  # 15, 16: sample frames from each clip
        root = cfg.DATASETS / cfg.SOURCE_ROOTS[source]
        vids = sorted(p for p in root.rglob("*") if p.suffix.lower() in cfg.VIDEO_EXTS)
        for v in vids:
            label = v.parent.name if source == "15" else v.stem.split("_")[0]
            yield from _iter_video_frames(v, label, "data")
    else:  # 2..11: rglob with the folder/filename heuristic
        root = cfg.DATASETS / source
        skip = cfg.SOURCE_EXCLUDE.get(source, ())
        for p in _images(root):
            if p.parent == root:
                continue  # stray images at the dataset root
            if skip and any(s in p.parts for s in skip):
                continue  # excluded subtree (e.g. source 9 ASL_dynamic)
            yield p, _heuristic_label(p), _split_of(p)


# --------------------------------------------------------------------------- #
# MediaPipe detection.                                                        #
# --------------------------------------------------------------------------- #
def make_landmarker(prefer_gpu: bool = True, quiet: bool = False):
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision as mpv

    last = None
    for name in (["GPU", "CPU"] if prefer_gpu else ["CPU"]):
        try:
            base = mpp.BaseOptions(
                model_asset_path=str(cfg.TASK_PATH),
                delegate=getattr(mpp.BaseOptions.Delegate, name),
            )
            det = mpv.HandLandmarker.create_from_options(
                mpv.HandLandmarkerOptions(
                    base_options=base,
                    running_mode=mpv.RunningMode.IMAGE,
                    num_hands=cfg.NUM_HANDS,
                    min_hand_detection_confidence=cfg.MIN_HAND_DETECTION_CONFIDENCE,
                    min_hand_presence_confidence=cfg.MIN_HAND_PRESENCE_CONFIDENCE,
                )
            )
            if not quiet:
                print(f"landmarker: {name} delegate")
            return det
        except Exception as exc:  # noqa: BLE001
            last = exc
            if not quiet:
                print(f"  {name} delegate unavailable: {repr(exc)[:120]}")
    raise RuntimeError(f"could not create HandLandmarker: {last}")


def _miss() -> dict:
    nan = [np.nan] * (cfg.NUM_KP * 3)
    return {"detected": 0, "handedness": "", "score": np.nan, "n": nan, "w": nan}


_FRAME_SEP = "\x00"   # ref for a video frame: f"{video_path}{_FRAME_SEP}{frame_idx}" (null can't be in a filename)


def _read_frame_rgb(ref: str):
    """RGB ndarray for an image path, or a 'videopath<sep>frameidx' video-frame ref.

    Returns None if the image/frame can't be read (recorded as a miss upstream).
    """
    if _FRAME_SEP in ref:
        import cv2
        vid, idx = ref.rsplit(_FRAME_SEP, 1)
        cap = cv2.VideoCapture(vid)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    from PIL import Image
    return np.asarray(Image.open(ref).convert("RGB"))


def _iter_video_frames(video_path, label, split):
    """Yield config.FRAMES_PER_VIDEO evenly-spaced frame refs for one video clip."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total <= 0:
        return
    n = cfg.FRAMES_PER_VIDEO
    if n >= total:
        idxs = range(total)
    elif n <= 1:
        idxs = [total // 2]
    else:
        idxs = [round(i * (total - 1) / (n - 1)) for i in range(n)]
    for i in idxs:
        yield f"{video_path}{_FRAME_SEP}{i}", label, split


def _detect(det, image_path) -> dict:
    import mediapipe as mp

    arr = _read_frame_rgb(str(image_path))
    if arr is None:
        return _miss()
    res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=arr))
    world = getattr(res, "hand_world_landmarks", None)
    if not res.hand_landmarks or not world:
        return _miss()
    lm, wl, hd = res.hand_landmarks[0], world[0], res.handedness[0][0]
    n = np.array([[p.x, p.y, p.z] for p in lm], np.float32).reshape(-1)
    w = np.array([[p.x, p.y, p.z] for p in wl], np.float32).reshape(-1)
    return {"detected": 1, "handedness": hd.category_name,
            "score": float(hd.score), "n": n, "w": w}


def _row(source, label, split, r, path="") -> list:
    if cfg.UPPERCASE_LABELS:
        label = str(label).upper()
    p = "" if _FRAME_SEP in str(path) else str(path)  # video frames aren't reloadable stills
    return [label, source, split, r["detected"], r["handedness"],
            r["score"], p, *r["n"], *r["w"]]


# --------------------------------------------------------------------------- #
# Progress + parallel workers.                                                #
# --------------------------------------------------------------------------- #
class _Prog:
    """tqdm bar when progress=True (nice in notebooks), else periodic prints."""

    def __init__(self, total, desc, progress):
        self.total, self.desc, self.n, self.det, self.bar = total, desc, 0, 0, None
        if progress:
            try:
                from tqdm.auto import tqdm
                self.bar = tqdm(total=total, desc=desc, unit="img")
            except Exception:  # noqa: BLE001
                self.bar = None

    def update(self, detected):
        self.n += 1
        self.det += detected
        if self.bar is not None:
            self.bar.update(1)
            if self.n % 200 == 0 or self.n == self.total:
                self.bar.set_postfix(detected=self.det)
        elif self.n % 5000 == 0:
            print(f"  {self.desc}: {self.n}/{self.total} ({self.det} detected)", flush=True)

    def close(self):
        if self.bar is not None:
            self.bar.close()


_WORKER_DET = None  # per-process landmarker for the multiprocessing path


def _init_worker():
    global _WORKER_DET
    _WORKER_DET = make_landmarker(prefer_gpu=False, quiet=True)


def _work(item) -> list:
    path, source, label, split = item
    try:
        r = _detect(_WORKER_DET, path)
    except Exception:  # noqa: BLE001  (corrupt/unreadable image -> recorded as a miss)
        r = _miss()
    return _row(source, label, split, r, path)


def _extract_serial(source, items, det, progress):
    rows, p = [], _Prog(len(items), f"{source} (serial)", progress)
    for path, label, split in items:
        try:
            r = _detect(det, path)
        except Exception as exc:  # noqa: BLE001
            print(f"  error {path}: {repr(exc)[:100]}", flush=True)
            r = _miss()
        rows.append(_row(source, label, split, r, path))
        p.update(r["detected"])
    p.close()
    return rows, p.det


def _extract_parallel(source, items, workers, progress):
    import multiprocessing as mpr

    tasks = [(str(path), source, label, split) for path, label, split in items]
    rows, p = [], _Prog(len(tasks), f"{source} ({workers}w)", progress)
    with mpr.get_context("spawn").Pool(workers, initializer=_init_worker) as pool:
        for row in pool.imap_unordered(_work, tasks, chunksize=cfg.CHUNKSIZE):
            rows.append(row)
            p.update(row[3])  # detected flag
    p.close()
    return rows, p.det


# --------------------------------------------------------------------------- #
# Extraction API.                                                             #
# --------------------------------------------------------------------------- #
def _folder14_frame() -> pd.DataFrame:
    """Folder 14 is already landmarks (x,y,z per joint) -> kp2d; no world 3D."""
    df = pd.read_csv(cfg.FOLDER14_CSV)
    coord = [f"{ax}{i}" for i in range(cfg.NUM_KP) for ax in ("x", "y", "z")]
    meta = pd.DataFrame({
        "label": df["label"].astype(str), "source": "14", "split": "data",
        "detected": 1, "handedness": "", "score": np.nan, "path": "",
    })
    kp2d = df[coord].astype(np.float32).set_axis(cfg.KP2D_COLS, axis=1)
    kp3d = pd.DataFrame(np.float32("nan"), index=df.index, columns=cfg.KP3D_COLS)
    return pd.concat([meta, kp2d, kp3d], axis=1)[cfg.ALL_COLS]


def _map_label(source, label):
    """Apply config.SOURCE_LABEL_MAP for this source (e.g. source 3: '12' -> 'M')."""
    return cfg.SOURCE_LABEL_MAP.get(source, {}).get(str(label), label)


def _in_scope(label) -> bool:
    """True if `label` is within config.CLASS_SCOPE (case-insensitive). An empty
    scope keeps every class."""
    return not cfg.CLASS_SCOPE or str(label).strip().lower() in cfg.CLASS_SCOPE


def _sample(items, limit):
    """Shuffle (config.SAMPLE_RANDOM) then cap to `limit`, keeping classes balanced.

    With config.SAMPLE_PER_CLASS, `limit` is a per-class cap. Otherwise `limit` is
    a total that is spread as evenly as possible across classes (round-robin), so
    the classes stay ~equal (each gets ceil/floor of limit/num_classes, capped by
    how many images it actually has). limit<=0 keeps all.
    """
    if cfg.SAMPLE_RANDOM:
        random.Random(cfg.SAMPLE_SEED).shuffle(items)
    if limit <= 0:
        return items

    buckets = defaultdict[Any, deque](deque)                 # class -> its items (shuffled order)
    for it in items:
        buckets[it[1]].append(it)                # it = (path, label, split)

    if cfg.SAMPLE_PER_CLASS:
        out = []
        for q in buckets.values():
            for _ in range(min(limit, len(q))):
                out.append(q.popleft())
        return out

    # total limit: round-robin one image per class per pass until we hit `limit`
    out, queues = [], list(buckets.values())
    while len(out) < limit and any(queues):
        for q in queues:
            if q:
                out.append(q.popleft())
                if len(out) >= limit:
                    break
    return out


def _sampled_items(source, limit):
    """(path, label, split) after label remap, class-scope filter, balanced sampling."""
    mapped = ((p, _map_label(source, lbl), split) for p, lbl, split in iter_source(source))
    return _sample([it for it in mapped if _in_scope(it[1])], limit)


def _extract_one(source, limit, det, workers, progress) -> pd.DataFrame:
    if source == "14":
        return _folder14_frame()
    items = _sampled_items(source, limit)
    mode = f"{workers} CPU workers" if workers >= 2 else "1 process"
    print(f"source {source}: {len(items)} images ({mode})", flush=True)
    if workers >= 2:
        rows, n_det = _extract_parallel(source, items, workers, progress)
    else:
        rows, n_det = _extract_serial(source, items, det, progress)
    print(f"source {source}: {n_det}/{len(rows)} detected", flush=True)
    return pd.DataFrame(rows, columns=cfg.ALL_COLS)


def image_items(sources, limit=None):
    """For image models (e.g. InceptionV3): (paths, labels) with the SAME class-scope
    filter, source-3 label remap, uppercase-normalization, and balanced sampling as
    the keypoint path -- but no MediaPipe. Folder 14 is skipped (keypoints, no image).
    """
    if isinstance(sources, str):
        sources = [sources]
    limit = cfg.DEFAULT_LIMIT if limit is None else limit
    paths, labels = [], []
    for s in sources:
        if s == "14" or s in cfg.VIDEO_SOURCES:  # no images to reload (csv / video frames)
            continue
        for p, lbl, _ in _sampled_items(s, limit):
            paths.append(str(p))
            labels.append(str(lbl).upper() if cfg.UPPERCASE_LABELS else str(lbl))
    return paths, labels


def extract_frame(sources, df=None, limit: int = 0, workers: int | None = None,
                  prefer_gpu: bool = True, progress: bool | None = None) -> pd.DataFrame:
    """Extract one or more sources into a DataFrame (in memory, no cache).

    If ``df`` is given, the new rows are appended onto it; any rows already in
    ``df`` for these sources are replaced (re-running a source updates it).
    """
    if isinstance(sources, str):
        sources = [sources]
    workers = cfg.DEFAULT_WORKERS if workers is None else workers
    progress = cfg.SHOW_PROGRESS if progress is None else progress

    # One shared landmarker for the serial path; parallel workers make their own.
    needs_model = any(s != "14" for s in sources)
    det = make_landmarker(prefer_gpu) if (needs_model and workers < 2) else None
    try:
        parts = [_extract_one(s, limit, det, workers, progress) for s in sources]
    finally:
        if det is not None:
            det.close()
    new = pd.concat(parts, ignore_index=True)

    if df is None or len(df) == 0:
        return new
    keep = df[~df["source"].isin(sources)]  # replace prior rows of these sources
    return pd.concat([keep, new], ignore_index=True)


def extract_dir(root, limit: int = 0, prefer_gpu: bool = True,
                progress: bool | None = None) -> pd.DataFrame:
    """Extract every <class>/<image> under an arbitrary directory into a DataFrame.

    Label = the image's immediate parent folder name. For external eval sets, e.g.
    evaluation_test/A/x.jpg -> label "A". Sampling (config.SAMPLE_*) still applies.
    """
    progress = cfg.SHOW_PROGRESS if progress is None else progress
    root = Path(root)
    name = root.name or "dir"
    items = _sample([(p, p.parent.name, "eval") for p in _images(root)
                     if _in_scope(p.parent.name)], limit)
    print(f"{name}: {len(items)} images (1 process)", flush=True)
    det = make_landmarker(prefer_gpu)
    try:
        rows, n_det = _extract_serial(name, items, det, progress)
    finally:
        det.close()
    print(f"{name}: {n_det}/{len(rows)} detected", flush=True)
    return pd.DataFrame(rows, columns=cfg.ALL_COLS)


def benchmark_workers(source="2", worker_counts=(0, 2, 4, 8), limit=400,
                      prefer_gpu=True) -> dict:
    """Time extraction of the same sample at each worker count. 0 = single
    process (GPU delegate if prefer_gpu). Prints img/s and rough RAM per run."""
    items = list(iter_source(source))[:limit]
    print(f"benchmark: {len(items)} images from source {source}\n")
    res = {}
    for w in worker_counts:
        det = make_landmarker(prefer_gpu, quiet=True) if w < 2 else None
        t = time.time()
        try:
            if w >= 2:
                _, n_det = _extract_parallel(source, items, w, progress=False)
            else:
                _, n_det = _extract_serial(source, items, det, progress=False)
        finally:
            if det is not None:
                det.close()
        dt = time.time() - t
        rate = len(items) / dt if dt else 0.0
        res[w] = rate
        ram = w * cfg.RAM_PER_WORKER_GB if w >= 2 else cfg.RAM_PER_WORKER_GB
        tag = f"{w} CPU workers" if w >= 2 else ("GPU serial" if prefer_gpu else "CPU serial")
        print(f"  {tag:14s}: {rate:6.1f} img/s  ({dt:4.1f}s)  ~{ram:.1f} GB RAM")
    return res


def selftest() -> None:
    df = extract_frame("2", limit=6, workers=0, progress=False)
    assert list(df.columns) == cfg.ALL_COLS
    det = df[df.detected == 1]
    assert det[cfg.KP2D_COLS].notna().to_numpy().all(), "detected kp2d must be finite"
    assert det[cfg.KP3D_COLS].notna().to_numpy().all(), "detected kp3d must be finite"

    df = extract_frame("14", df, workers=0, progress=False)  # incremental append
    assert set(df.source) == {"2", "14"}
    f14 = df[df.source == "14"]
    assert f14[cfg.KP2D_COLS].notna().to_numpy().all(), "folder-14 kp2d must be finite"
    assert f14[cfg.KP3D_COLS].isna().to_numpy().all(), "folder-14 kp3d must be NaN"

    n2 = int((df.source == "2").sum())
    df = extract_frame("2", df, limit=6, workers=0, progress=False)  # re-run updates
    assert int((df.source == "2").sum()) == n2, "re-run should replace, not duplicate"
    print(f"selftest OK: {len(df)} rows, sources={sorted(df.source.unique())}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="+", help="default: all image sources + 14")
    ap.add_argument("--limit", type=int, default=cfg.DEFAULT_LIMIT,
                    help="cap images per source (per class if config.SAMPLE_PER_CLASS); 0 = all")
    ap.add_argument("--workers", type=int, default=None,
                    help="CPU worker processes (>=2). Default from config.DEFAULT_WORKERS")
    ap.add_argument("--cpu", action="store_true", help="force CPU delegate (serial)")
    ap.add_argument("--benchmark", action="store_true", help="compare worker counts")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if args.benchmark:
        src = (args.sources or ["2"])[0]
        benchmark_workers(source=src, limit=args.limit or 400, prefer_gpu=not args.cpu)
        return

    sources = args.sources or (cfg.IMAGE_SOURCES + ["14"])
    df = extract_frame(sources, limit=args.limit, workers=args.workers,
                       prefer_gpu=not args.cpu)
    print(df[["label", "source", "split", "detected"]].head())
    print("rows:", len(df), "| detected:", int(df.detected.sum()))


if __name__ == "__main__":
    main()
