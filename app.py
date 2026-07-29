"""Real-time ASL letter recognition from the webcam.

Detects the hand with MediaPipe (hand_landmarker.task) and runs whichever trained
model --weights points at (default: config.MODEL_WEIGHTS). Every model kind works:

  * .joblib               -> MLP, keypoint features (hand_feature)
  * .keras, keypoints     -> mini_cnn: hand_feature -> softmax
  * .keras, image         -> cnn / inceptionv3: the hand is cropped from the frame
                             (using the landmark bbox) and fed as an image
  * .keras, image+kp      -> mini_cnn+cnn: both the crop and the keypoints are fed

    python app.py                 # predict letters (default, camera 1)
    python app.py --model         # landmarks only, no letter prediction
    python app.py --camera 0      # pick a different webcam
    python app.py --weights mini_cnn/asl_mini_cnn_both.keras
    python app.py --landmarker path/to/hand_landmarker.task

Controls: press 'q' or ESC to quit.

Requires: mediapipe, opencv-python, numpy, pandas, joblib; tensorflow for .keras models.
    pip install mediapipe opencv-python numpy pandas joblib tensorflow
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# Shared feature builder (config.FEATURES / FLIP_LEFT_TO_RIGHT), same as training.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "neural_network"))
import config as cfg
from extract import hand_feature

# MediaPipe Hands 21-joint connectivity.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm edge
)

# Where to look for the MediaPipe landmark model if --landmarker isn't given.
LANDMARKER_CANDIDATES = [
    Path(__file__).resolve().parent / "hand_landmarker.task",
    Path.home() / "datasets" / "hand_landmarker.task",
    Path.home() / "hand-pose" / "models" / "hand_landmarker.task",
]

# Where to look for the trained classifier if --weights isn't given.
WEIGHTS_CANDIDATES = [
    Path(cfg.MODEL_WEIGHTS),                          # model chosen in config.MODEL
    Path(__file__).resolve().parent / "asl_mlp.joblib",
]


def locate(explicit: str | None, candidates: list[Path], what: str) -> str:
    if explicit:
        if not Path(explicit).is_file():
            raise SystemExit(f"{what} not found: {explicit}")
        return explicit
    for cand in candidates:
        if cand.is_file():
            return str(cand)
    raise SystemExit(f"could not find {what}; pass its path explicitly")


def make_landmarker(model_path: str, num_hands: int):
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def load_classifier(weights_path: str) -> dict:
    """Build a model bundle: kind (joblib/keras), the model, class list, image size.

    For .keras the input type is inferred from the model's input shape:
    two inputs -> concat (image+kp); (None,H,W,3) -> image; (None,feat) -> keypoints.
    """
    if weights_path.endswith(".joblib"):
        import joblib
        return {"kind": "joblib", "model": joblib.load(weights_path),
                "type": "kp", "classes": None, "size": None, "channels": None}

    os.environ["CUDA_VISIBLE_DEVICES"] = ""               # TF shares process w/ MediaPipe -> CPU
    import tensorflow as tf
    model = tf.keras.models.load_model(weights_path)
    classes = json.loads(Path(os.path.splitext(weights_path)[0] + "_classes.json").read_text())
    shapes = model.input_shape
    if isinstance(shapes, list):                          # two inputs -> image + keypoints
        img = next(s for s in shapes if len(s) == 4)
        kind, size, ch = "concat", img[1], img[3]
    elif len(shapes) == 4:                                # (None,H,W,C) -> image model
        kind, size, ch = "image", shapes[1], shapes[3]
    else:                                                 # (None, feat) -> keypoint model
        kind, size, ch = "kp", None, None
    return {"kind": "keras", "model": model, "type": kind, "classes": classes,
            "size": size, "channels": ch}


def crop_hand(frame: np.ndarray, pts, size: int, channels: int = 3, margin: float = 0.3) -> np.ndarray:
    """Square hand crop from the landmark bbox (float32), for image models.

    channels=1 returns a grayscale (H,W,1) crop (mini_cnn+cnn); 3 returns RGB.
    """
    h, w = frame.shape[:2]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    mx = int((max(xs) - min(xs)) * margin) + 1
    my = int((max(ys) - min(ys)) * margin) + 1
    x0, y0 = max(0, min(xs) - mx), max(0, min(ys) - my)
    x1, y1 = min(w, max(xs) + mx), min(h, max(ys) + my)
    roi = frame[y0:y1, x0:x1]
    if roi.size == 0:
        roi = frame
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)            # training decoded images as RGB
    roi = cv2.resize(roi, (size, size)).astype(np.float32)
    if channels == 1:                                     # grayscale image branch
        roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)[..., None]
    return roi


def predict_letter(bundle, world_landmarks, hand_landmarks, handedness, frame, pts) -> tuple[str, float]:
    """Classify one hand with whatever model kind is loaded (kp / image / concat)."""
    def kp_feats():                                       # exact training features (FEATURES/flip)
        xyz = np.array([[p.x, p.y, p.z] for p in world_landmarks], dtype=np.float32)
        xy = np.array([[p.x, p.y, p.z] for p in hand_landmarks], dtype=np.float32)
        return hand_feature(xyz, xy, handedness)[None]

    if bundle["kind"] == "joblib":                        # numpy MLP: label + proba directly
        feats = kp_feats()
        return bundle["model"].predict(feats)[0], float(bundle["model"].predict_proba(feats).max())

    model, kind = bundle["model"], bundle["type"]         # keras: softmax probs -> class label
    if kind == "kp":
        inp = kp_feats()
    elif kind == "image":
        inp = crop_hand(frame, pts, bundle["size"], bundle["channels"])[None]
    else:                                                 # concat: match model's input order
        img = crop_hand(frame, pts, bundle["size"], bundle["channels"])[None]
        feats = kp_feats()
        inp = [img if len(t.shape) == 4 else feats for t in model.inputs]
    probs = np.asarray(model(inp, training=False))[0]
    i = int(probs.argmax())
    return bundle["classes"][i], float(probs[i])


def draw_hands(frame: np.ndarray, result, bundle) -> None:
    h, w = frame.shape[:2]
    clean = frame.copy()                                  # crop images before overlays are drawn
    hands = getattr(result, "hand_landmarks", None) or []
    world = getattr(result, "hand_world_landmarks", None) or []
    handedness = getattr(result, "handedness", None) or []

    for idx, lm in enumerate(hands):
        pts = [(int(p.x * w), int(p.y * h)) for p in lm]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 220, 0), 2)
        for i, (x, y) in enumerate(pts):
            color = (0, 220, 255) if i == 0 else (32, 32, 255)  # wrist vs joints (BGR)
            cv2.circle(frame, (x, y), 4, color, -1)

        if bundle is not None and idx < len(world):
            hd = handedness[idx][0].category_name if idx < len(handedness) and handedness[idx] else ""
            letter, proba = predict_letter(bundle, world[idx], lm, hd, clean, pts)
            cv2.putText(frame, f"{letter} {proba:.2f}",
                        (pts[0][0] - 10, pts[0][1] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
        elif idx < len(handedness) and handedness[idx]:
            cat = handedness[idx][0]
            # The frame is mirrored (cv2.flip) before detection, so MediaPipe's
            # Left/Right is reversed relative to the user's real hand. Swap it back.
            true_name = {"Left": "Right", "Right": "Left"}.get(
                cat.category_name, cat.category_name)
            cv2.putText(frame, f"{true_name} {cat.score:.2f}",
                        (pts[0][0] - 10, pts[0][1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="store_true",
                        help="landmarks only: skip letter prediction")
    parser.add_argument("--weights", default=None,
                        help="path to trained weights (.joblib or .keras); default config.MODEL_WEIGHTS")
    parser.add_argument("--landmarker", default=None,
                        help="path to hand_landmarker.task")
    parser.add_argument("--camera", type=int, default=1, help="webcam index")
    parser.add_argument("--num-hands", type=int, default=2)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    landmarker_path = locate(args.landmarker, LANDMARKER_CANDIDATES, "hand_landmarker.task")
    print(f"landmarker: {landmarker_path}")

    bundle = None
    if not args.model:
        weights_path = locate(args.weights, WEIGHTS_CANDIDATES, "model weights")
        bundle = load_classifier(weights_path)
        print(f"classifier: {weights_path}  ({bundle['kind']}/{bundle['type']})")
    else:
        print("landmarks-only mode (no letter prediction)")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {args.camera}")

    landmarker = make_landmarker(landmarker_path, args.num_hands)
    start = time.time()
    prev = start
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("failed to read frame")
                break

            frame = cv2.flip(frame, 1)  # mirror for a natural selfie view
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.time() - start) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            draw_hands(frame, result, bundle)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
            prev = now
            cv2.putText(frame, f"{fps:4.1f} FPS", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("ASL Recognition (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()


if __name__ == "__main__":
    main()
