"""Editable constants for keypoint extraction. Tweak values here, not in dataloader.py."""

import string
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASETS = BASE_DIR / "datasets"
TASK_PATH = BASE_DIR / "hand_landmarker.task"

# --- Features (what the model eats; changing these means retraining) --------- #
# "2d"      = image x,y landmarks only (42 dims)
# "2.5d"    = image x,y + relative depth z (63 dims)
# "3d"      = jazz world landmarks, global min-max normalized to [0,1] per hand (63 dims)
# "2d+3d"   = 42 + 63 (105 dims)
# "2.5d+3d" = 63 + 63 (126 dims)
FEATURES = "2d+3d"
# Mirror left hands onto right-hand chirality (negate x) so both map to one shape.
FLIP_LEFT_TO_RIGHT = True

# The single keypoint cache shared by EVERY run (run.py --cache, experiments.ipynb).
# It stores RAW jazz landmarks, not features, so switching FEATURES does not
# invalidate it and all feature modes are compared on identical underlying data.
EXTRACT_CACHE = BASE_DIR / "extract_cache.npz"

# --- Model selection -------------------------------------------------------- #
# Pick which model the pipeline uses. Each model lives in its own folder with its
# code AND its saved weights beside it. To add a model, drop a folder here.
MODEL = "mlp"                       # active model key from MODELS below
# "input" = "keypoints" (extract.py -> MediaPipe), "image" (raw images), or
#           "image+keypoints" (both: extract emits paths + keypoints, trainer also
#           loads the images). "train" = script run.py launches (repo-root relative).
MODELS = {
    "mlp":         {"dir": "neural_network", "weights": "asl_mlp.joblib",        "input": "keypoints",       "train": "train.py"},
    "mini_cnn":    {"dir": "mini_cnn",       "weights": "asl_mini_cnn.keras",    "input": "keypoints",       "train": "mini_cnn/asl_mini_cnn.py"},
    "cnn":         {"dir": "cnn",            "weights": "asl_cnn.keras",         "input": "image",           "train": "cnn/asl_cnn.py"},
    "inceptionv3": {"dir": "InceptionV3",    "weights": "asl_inceptionv3.keras", "input": "image",           "train": "InceptionV3/asl_inceptionv3.py"},
    "mini_cnn+cnn":{"dir": "mini_cnn+cnn",   "weights": "asl_mini_cnn_cnn.keras","input": "image+keypoints", "train": "mini_cnn+cnn/asl_mini_cnn_cnn.py"},
}
MODEL_DIR = BASE_DIR / MODELS[MODEL]["dir"]                 # active model's folder
# Any model that consumes keypoints gets the feature-mode suffix so the modes never
# overwrite each other (e.g. asl_mlp_both.joblib). Pure image models take images
# directly, so no suffix.
_w = Path(MODELS[MODEL]["weights"])
_fsuffix = f"_{FEATURES}" if "keypoints" in MODELS[MODEL].get("input", "") else ""
MODEL_WEIGHTS = MODEL_DIR / f"{_w.stem}{_fsuffix}{_w.suffix}"

# --- MediaPipe hand landmarker (matches the jazz pipeline) ------------------ #
NUM_HANDS = 1
MIN_HAND_DETECTION_CONFIDENCE = 0.3
MIN_HAND_PRESENCE_CONFIDENCE = 0.3

# --- Extraction defaults ---------------------------------------------------- #
DEFAULT_WORKERS = 0     # 0 = single process on GPU delegate; N>=2 = N CPU processes
CHUNKSIZE = 32          # images dispatched per worker at a time (parallel mode)
SHOW_PROGRESS = True
RAM_PER_WORKER_GB = 0.9  # each worker loads its own MediaPipe+TF (~0.9 GB). Used
#                          only to warn/estimate: workers * this must fit in RAM.

# --- Sampling (applied per source before detection; used everywhere) -------- #
DEFAULT_LIMIT = 5000        # images kept per source, 0 = all. CLI --limit overrides this.
SAMPLE_RANDOM = True     # True = shuffle before the limit (random sample); False = first-N in walk order
SAMPLE_PER_CLASS = False # True = limit is images PER CLASS; False = total per source, spread ~evenly across classes
SAMPLE_SEED = 0          # seed for the random sample, so a run is reproducible

# --- Training (train.py) ---------------------------------------------------- #
EPOCHS = 100             # training passes over the data
BATCH_SIZE = 256
HIDDEN_LAYERS = (256, 128)   # MLP hidden layer sizes (ReLU between each)
BATCH_NORM = True        # BatchNorm1d after each hidden ReLU (before dropout)
DROPOUT = 0.3            # dropout after each hidden ReLU (0 = off); train-time only
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.2          # fraction held out for the per-epoch validation accuracy

# How many times a configuration is retrained before anything is reported.
# The train/val split is fixed (seed 0), but weight init, batch shuffling and the
# dropout masks are not, so one run is a single draw. Repeating and reporting
# mean +/- std is what tells you whether a gap between two setups is real or noise.
# Honored by experiments.ipynb and by train.py (so run.py too); train.py keeps the
# MEDIAN run's weights, so the accuracy it prints is one a retrain can reproduce.
# 1 = no repeats.
ITERATIONS = 5

# --- MLP model-size experiment ---------------------------------------------- #
# EXPERIMENT_MODE = True runs the MLP at the size named by EXPERIMENT: it OVERRIDES
# HIDDEN_LAYERS, forces FEATURES to 3d, and redirects ALL outputs (weights, plots,
# _test.txt, confusion PNGs, the _runs.joblib experiments_size.ipynb reads) to
# experiments/<EXPERIMENT>/ instead of the model's own folder, so a sweep never
# clobbers your real weights. Keep --sources/--limit (or a --cache) fixed across the
# sweep, and then size is the only thing that differs between runs.
MLP_SIZES = {
    "h32":          (32,),
    "h64":          (64,),
    "h128":         (128,),
    "h256":         (256,),
    "h512":         (512,),
    "h128_64":      (128, 64),
    "h256_128":     (256, 128),
    "h512_256":     (512, 256),
    "h256_128_64":  (256, 128, 64),
    "h512_256_128": (512, 256, 128),
    # --- bigger: eval F1 was still climbing at (512,256,128), so probe wider/deeper ---
    "h1024":          (1024,),
    "h1024_512":      (1024, 512),
    "h512_512_256":   (512, 512, 256),
    "h1024_512_256":  (1024, 512, 256),
    "h1024_1024_512": (1024, 1024, 512),
}
EXPERIMENT_MODE = True# toggle: False = normal run (saves to neural_network/),
#                          True = run the size sweep (saves to experiments/<EXPERIMENT>/)
EXPERIMENT = "h1024_1024_512"      # which MLP_SIZES entry to use when EXPERIMENT_MODE is on

if EXPERIMENT_MODE:
    if EXPERIMENT not in MLP_SIZES:
        raise ValueError(f"EXPERIMENT {EXPERIMENT!r} not in MLP_SIZES {list(MLP_SIZES)}")
    HIDDEN_LAYERS = MLP_SIZES[EXPERIMENT]
    FEATURES = "3d"          # the size sweep varies size ONLY; 3d won the feature sweep
    MODEL_DIR = BASE_DIR / "experiments" / EXPERIMENT
    MODEL_WEIGHTS = MODEL_DIR / f"{_w.stem}_{EXPERIMENT}_{FEATURES}{_w.suffix}"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)  # train.py (MLP) doesn't create its own dir

# --- InceptionV3 fine-tuning (image model only) ----------------------------- #
# Phase 1 trains just the new head with the backbone frozen (feature extraction).
# If FINE_TUNE, phase 2 then unfreezes the backbone and keeps training at a LOW
# LR so the pretrained ImageNet weights adapt to hands without being wrecked.
FINE_TUNE = True         # False = feature-extraction only (frozen backbone)
FINE_TUNE_EPOCHS = 50    # extra epochs with the backbone unfrozen (phase 2 does the real learning)
FINE_TUNE_LR = 1e-5      # small LR for phase 2 (BatchNorm layers stay frozen)
FINE_TUNE_UNFREEZE = 50  # unfreeze the top N backbone layers (0 = the whole backbone)

# --- Testing (test.py / run.py) --------------------------------------------- #
TEST_LIMIT = 200         # images per source for the sampled test (0 = all)
TEST_SEED = SAMPLE_SEED + 1  # different sample than training, so the test is fresh
# The ONE held-out evaluation set: test_set/<left|right>/<CLASS>/<frames> (frames 1,
# 6, 11, ... of signlanguagetestdata). Every reported score comes from here, so runs
# are comparable. Frames sitting directly under left/ or right/ have no class folder,
# hence no label, and are dropped by CLASS_SCOPE.
TEST_SET_DIR = BASE_DIR / "test_set"
EVAL_LIMIT = 0           # images per class from TEST_SET_DIR (0 = all)

# --- Labels ----------------------------------------------------------------- #
UPPERCASE_LABELS = True   # normalize every label to uppercase so 'a' and 'A' are one class

# Per-source label remap: {source: {raw_label: canonical_label}}. Sources not
# listed keep their labels as-is. Source 3 names its class folders 0-27; 0-25 map
# to A-Z (folders 26,27 are non-letters and get dropped by CLASS_SCOPE).
SOURCE_LABEL_MAP = {
    "3": {str(i): string.ascii_uppercase[i] for i in range(26)},
}

# --- Class scope ------------------------------------------------------------ #
# Only images whose label is in this set are kept (matched case-insensitively);
# an empty set keeps everything. J and Z are dynamic (motion): a still frame can't
# express them, so they are out of scope and the task is the 24 static letters.
# Non-letter folders (digits, del, space) get dropped too.
CLASS_SCOPE = frozenset(string.ascii_lowercase) - {"j", "z"}

# --- Sources ---------------------------------------------------------------- #
# Folder 1 is intentionally excluded. 14 is a landmark CSV (kp2d only).
# 15, 16 are VIDEO sources (frames sampled at runtime, see VIDEO_* below).
IMAGE_SOURCES = ["2", "3", "4", "5", "6", "8", "9", "10", "11", "12", "13", "15", "16"]
FOLDER14_CSV = DATASETS / "14/asl_landmarks_final.csv"

# For these sources only the given subtree (relative to DATASETS) is used.
SOURCE_ROOTS = {
    "12": "12/extracted/root/Root/Type_01_(Raw_Gesture)",  # only the Type_01 images
    "13": "13/output/dataset",                             # only <class>/lit (below)
    "15": "15/SigNN Video Data",                           # <LETTER>/<n>.avi (J, Z)
    "16": "16/video_data",                                 # <LETTER>_<L|R>.mp4 (all letters)
}
FOLDER13_STYLE = "lit"  # folder 13: use only this render style under each class

# --- Video sources ---------------------------------------------------------- #
# Frames are sampled from each clip (FRAMES_PER_VIDEO evenly spaced) and run through
# MediaPipe like images. Image-only models (cnn, inceptionv3) skip these, since a
# sampled video frame isn't a reloadable still file.
VIDEO_SOURCES = {"15", "16"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
FRAMES_PER_VIDEO = 12   # evenly-spaced frames pulled per clip

# Skip any image whose path contains one of these parts, per source. E.g. source
# 9 has an ASL_dynamic (video-frame) subtree we don't want; keep only SignAlphaSet.
SOURCE_EXCLUDE = {
    "9": ("ASL_dynamic",),
}

# Folder/filename parts that are split/dataset containers, not class labels.
SPLIT_CONTAINERS = {
    "train", "test", "valid", "validation", "data",
    "asl_alphabet_train", "asl_alphabet_test", "asl_alphabet_dataset",
}

NUM_KP = 21
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# --- Output schema: 2D image landmarks (n*) + 3D world landmarks (w*) -------- #
KP2D_COLS = [f"n{k}" for k in range(NUM_KP * 3)]
KP3D_COLS = [f"w{k}" for k in range(NUM_KP * 3)]
# `path` = source image path, kept so image+keypoint models can pair pixels with
# the detected keypoints ("" for folder-14 CSV rows, which have no image).
META_COLS = ["label", "source", "split", "detected", "handedness", "score", "path"]
ALL_COLS = META_COLS + KP2D_COLS + KP3D_COLS
HAND_MAP = {"Left": 0, "Right": 1, "": -1}
