"""Orchestrator: ONE command trains whichever model config.MODEL selects.

    python run.py

It reads config.MODEL and dispatches to that model's own trainer (config.MODELS
[MODEL]["train"]), so every model runs the same way -- you only change config.MODEL:

  * keypoint models (mlp, mini_cnn) and image+keypoint models (mini_cnn+cnn):
    extract.py (MediaPipe, GPU) is piped in memory into the model's trainer. Keypoints
    (and, for the concat model, the matching image paths) never touch disk. Two
    processes so training gets a clean CUDA context (no MediaPipe in it).
  * image models (cnn, inceptionv3): no extraction -- the trainer loads raw images
    itself (dataloader.image_items), so no MediaPipe/keypoint step runs at all.

Then an offline test runs (own process). Alongside the weights, each run writes:
a <stem>_test.txt report (with a runtime breakdown appended here), accuracy/loss
history PNGs (from the trainer), and a confusion-matrix PNG per test set (from
test.py). Weights and images all go to config.MODEL_DIR.

    python run.py                       # train the config.MODEL model + test
    python run.py --sources 6 --limit 1000
    python run.py --no-test
    python run.py --cache               # reuse (or create) a keypoint-extraction cache

By default extraction runs fresh every time (nothing cached). --cache reuses a saved
extraction if present, else creates it -- handy for architecture/feature sweeps. There
is exactly ONE cache, extract_cache.npz at the repo root, and it holds the RAW jazz
landmarks rather than feature vectors, so every FEATURES mode reads the same file and
every experiment is scored on byte-identical data. Delete it only if you change
--sources, --limit, or config.CLASS_SCOPE. Image models (cnn, inceptionv3) ignore it.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg


def _fmt(seconds):
    """Compact h/m/s, or 'n/a' for a stage that didn't run."""
    if seconds is None:
        return "n/a"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return (f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s" if m else f"{s}s") + f" ({seconds:.1f}s)"


def _append_runtime(extract_s, train_s, test_s, total_s, is_image):
    """Append a runtime section to the model's <stem>_test.txt report."""
    report = Path(cfg.MODEL_DIR) / f"{Path(cfg.MODEL_WEIGHTS).stem}_test.txt"
    kp = "n/a (image model)" if is_image else _fmt(extract_s)
    block = ("\n=== runtime ===\n"
             f"images -> keypoints : {kp}\n"
             f"training            : {_fmt(train_s)}\n"
             f"testing             : {_fmt(test_s)}\n"
             f"total               : {_fmt(total_s)}\n")
    print(block)
    if report.is_file():
        with report.open("a") as fh:
            fh.write(block)
        print(f"runtime appended -> {report}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sources", nargs="+", default=cfg.IMAGE_SOURCES,
                    help="sources for keypoint extraction (ignored by image models)")
    ap.add_argument("--limit", type=int, default=cfg.DEFAULT_LIMIT,
                    help="images per source for keypoint extraction (ignored by image models)")
    ap.add_argument("--test-limit", type=int, default=cfg.TEST_LIMIT,
                    help="images per source for the offline sampled test (0 = same as --limit)")
    ap.add_argument("--no-test", action="store_true", help="skip the offline test stage")
    ap.add_argument("--cache", nargs="?", default=None, const=str(cfg.EXTRACT_CACHE),
                    help="reuse/create the shared keypoint cache (default: fresh every run)")
    a = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    py = sys.executable
    model = cfg.MODELS[cfg.MODEL]
    is_image = model["input"] == "image"
    trainer = os.path.join(here, *model["train"].split("/"))
    print(f"model: {cfg.MODEL} ({model['input']}) -> {trainer}")

    t0 = time.time()
    extract_s = None                                     # image models have no extraction
    if not is_image:
        ex = [py, os.path.join(here, "extract.py"),
              "--sources", *a.sources, "--limit", str(a.limit)]
        # a 0-byte/partial cache (e.g. an interrupted run) is NOT a valid hit
        if a.cache and os.path.isfile(a.cache) and os.path.getsize(a.cache) > 0:
            # cache hit: skip MediaPipe entirely, feed the saved .npz to the trainer
            print(f"reusing cached extraction -> {a.cache}")
            with open(a.cache, "rb") as fh:
                r = subprocess.run([py, trainer], stdin=fh)
            extract_s = 0.0
            if r.returncode:
                sys.exit(f"{model['train']} failed ({r.returncode})")
        elif a.cache:
            # cache miss: extract once to build it (first --cache run always extracts).
            # Write to a .tmp and only rename on success, so an interrupted extract
            # never leaves a corrupt/0-byte cache that a later run would reuse.
            print(f"no valid cache at {a.cache}; extracting once to build it")
            tmp = a.cache + ".tmp"
            with open(tmp, "wb") as fh:
                p1 = subprocess.run(ex, stdout=fh)
            extract_s = time.time() - t0
            if p1.returncode or os.path.getsize(tmp) == 0:
                os.remove(tmp)
                sys.exit(f"extract.py failed ({p1.returncode}); cache not written")
            os.replace(tmp, a.cache)
            print(f"cached extraction -> {a.cache}")
            with open(a.cache, "rb") as fh:
                r = subprocess.run([py, trainer], stdin=fh)
            if r.returncode:
                sys.exit(f"{model['train']} failed ({r.returncode})")
        else:
            # default: extract (MediaPipe, GPU) -> trainer (GPU), in-memory pipe, no disk.
            # train.py reads all of stdin before it computes, so extract finishes first:
            # p1's exit ~= extraction done, and the remainder is training wall time.
            p1 = subprocess.Popen(ex, stdout=subprocess.PIPE)
            p2 = subprocess.Popen([py, trainer], stdin=p1.stdout)
            p1.stdout.close()            # let extract get SIGPIPE if the trainer dies
            p1.wait()
            extract_s = time.time() - t0
            p2.wait()
            if p1.returncode:
                sys.exit(f"extract.py failed ({p1.returncode})")
            if p2.returncode:
                sys.exit(f"{model['train']} failed ({p2.returncode})")
    else:
        # image model: the trainer loads images itself -- no extraction, no MediaPipe
        r = subprocess.run([py, trainer])
        if r.returncode:
            sys.exit(f"{model['train']} failed ({r.returncode})")
    train_s = time.time() - t0 - (extract_s or 0.0)

    # stage 3: offline test (own process): the trainer's real val split + evaluation_test
    test_s = None
    if not a.no_test:
        t_test = time.time()
        subprocess.run([py, os.path.join(here, "test.py")], check=True)
        test_s = time.time() - t_test

    _append_runtime(extract_s, train_s, test_s, time.time() - t0, is_image)
    print("done ->", cfg.MODEL_WEIGHTS)


if __name__ == "__main__":
    main()
