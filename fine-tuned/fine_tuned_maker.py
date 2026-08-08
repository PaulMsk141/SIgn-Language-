#!/usr/bin/env python3
"""Capture up to 1000 fine-tuning images per hand for each ASL alphabet letter."""

from __future__ import annotations

import argparse
import re
import shutil
import string
import subprocess
import sys
import time
from pathlib import Path

import cv2


LETTERS = string.ascii_uppercase
HANDS = ("left", "right")
MAX_IMAGES = 1000
WINDOW_NAME = "ASL Fine-Tune Maker"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture A1..A1000 through Z1..Z1000 per hand from a camera."
    )
    parser.add_argument(
        "--camera",
        default="obs",
        help='Camera name or device index (default: "obs").',
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="List macOS AVFoundation cameras and exit.",
    )
    parser.add_argument(
        "--letter",
        choices=LETTERS,
        default="A",
        help="Letter selected when the program starts (default: A).",
    )
    parser.add_argument(
        "--hand",
        choices=HANDS,
        default="right",
        help="Hand selected when the program starts (default: right).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Folder containing the left/ and right/ directories "
        "(default: this script's folder).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=MAX_IMAGES,
        help=f"Images kept per letter per hand (default: {MAX_IMAGES}).",
    )
    parser.add_argument(
        "--burst-fps",
        type=float,
        default=10.0,
        help="Images saved per second while burst capture is on (default: 10).",
    )
    parser.add_argument(
        "--burst-delay",
        type=float,
        default=2.0,
        help="Seconds to get into position before a burst starts (default: 2).",
    )
    return parser.parse_args()


def avfoundation_video_devices() -> dict[int, str]:
    """Return macOS video devices reported by FFmpeg."""
    if sys.platform != "darwin" or shutil.which("ffmpeg") is None:
        return {}

    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-f",
                "avfoundation",
                "-list_devices",
                "true",
                "-i",
                "",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}

    devices: dict[int, str] = {}
    reading_video_devices = False
    for line in result.stderr.splitlines():
        if "AVFoundation video devices:" in line:
            reading_video_devices = True
            continue
        if "AVFoundation audio devices:" in line:
            break
        if not reading_video_devices:
            continue

        match = re.search(r"\]\s+\[(\d+)\]\s+(.+)$", line)
        if match:
            devices[int(match.group(1))] = match.group(2).strip()
    return devices


def resolve_camera(camera_value: str) -> tuple[int, str, dict[int, str]]:
    """Resolve a numeric index or a partial AVFoundation camera name.

    Indices are positions in the list macOS returns right now, so the same
    camera can move when another device appears or drops out. The device list
    is returned alongside the match to make such a shift visible.
    """
    try:
        index = int(camera_value)
    except ValueError:
        index = -1
    else:
        return index, f"camera {index}", {}

    devices = avfoundation_video_devices()
    requested_name = camera_value.casefold()
    for device_index, device_name in devices.items():
        if requested_name in device_name.casefold():
            return device_index, device_name, devices

    available = ", ".join(
        f"{device_index}: {device_name}"
        for device_index, device_name in devices.items()
    )
    if not available:
        available = "none detected"
    raise RuntimeError(
        f'Could not find a camera matching "{camera_value}". '
        f"Available cameras: {available}. Start OBS Virtual Camera first."
    )


def open_camera(index: int):
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    camera = cv2.VideoCapture(index, backend)
    if not camera.isOpened() and backend != cv2.CAP_ANY:
        camera.release()
        camera = cv2.VideoCapture(index)
    return camera


def ensure_letter_folders(output_dir: Path) -> None:
    for hand in HANDS:
        for letter in LETTERS:
            (output_dir / hand / letter).mkdir(parents=True, exist_ok=True)


def image_path(output_dir: Path, hand: str, letter: str, slot: int) -> Path:
    return output_dir / hand / letter / f"{letter}{slot}.jpg"


def used_slots(output_dir: Path, hand: str, letter: str, max_images: int) -> set[int]:
    """Slot numbers already on disk for one hand's letter."""
    pattern = re.compile(rf"^{letter}(\d+)\.jpg$")
    slots: set[int] = set()
    letter_dir = output_dir / hand / letter
    if not letter_dir.is_dir():
        return slots

    for entry in letter_dir.iterdir():
        match = pattern.match(entry.name)
        if match:
            slot = int(match.group(1))
            if 1 <= slot <= max_images:
                slots.add(slot)
    return slots


def next_available_slot(slots: set[int], max_images: int) -> int | None:
    """Lowest free slot, so deleted images are refilled before new ones are added."""
    for slot in range(1, max_images + 1):
        if slot not in slots:
            return slot
    return None


def draw_status(
    frame,
    camera_label: str,
    hand: str,
    letter: str,
    saved: int,
    max_images: int,
    burst_state: str,
    message: str,
):
    display = frame.copy()
    height, width = display.shape[:2]

    cv2.rectangle(display, (0, 0), (width, 116), (20, 20, 20), -1)
    cv2.putText(
        display,
        f"{hand.upper()} hand    Letter: {letter}    "
        f"Images: {saved}/{max_images}    Burst: {burst_state}",
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        display,
        "A-Z: select | SPACE: capture | . : burst | / : hand | BKSP: delete last | "
        "[ ]: navigate | ESC: quit",
        (20, 73),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        display,
        message,
        (20, 103),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (80, 220, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        display,
        f"Source: {camera_label}    Preview and saved image are not mirrored",
        (20, height - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return display


def main() -> int:
    args = parse_args()

    if args.list_cameras:
        devices = avfoundation_video_devices()
        if not devices:
            print("No cameras detected. Start OBS Virtual Camera and try again.")
            return 1
        for device_index, device_name in devices.items():
            print(f"{device_index}: {device_name}")
        return 0

    max_images = max(1, args.max_images)
    burst_interval = 1.0 / args.burst_fps if args.burst_fps > 0 else 0.0

    output_dir = args.output.expanduser().resolve()
    ensure_letter_folders(output_dir)

    try:
        camera_index, camera_name, devices = resolve_camera(args.camera)
    except RuntimeError as error:
        print(error)
        return 1

    if devices:
        listing = ", ".join(f"{i}: {name}" for i, name in devices.items())
        print(f"Cameras now: {listing}")

    camera_label = f"{camera_name} (device {camera_index})"
    print(f"Opening {camera_label}...")
    camera = open_camera(camera_index)
    if not camera.isOpened():
        print(
            f"Could not open {camera_name}. In OBS, click Start Virtual Camera "
            "and then run this script again."
        )
        return 1

    selected_index = LETTERS.index(args.letter)
    letter = LETTERS[selected_index]
    hand = args.hand
    slots = used_slots(output_dir, hand, letter, max_images)
    message = "SPACE for one image, . for burst capture."
    bursting = False
    burst_starts_at = 0.0
    next_burst_at = 0.0
    next_rescan_at = 0.0

    def save_next(frame) -> str:
        slot = next_available_slot(slots, max_images)
        if slot is None:
            return f"{hand}/{letter} already has {max_images} images."

        destination = image_path(output_dir, hand, letter, slot)
        if not cv2.imwrite(str(destination), frame):
            return f"Failed to save {hand}/{letter}/{destination.name}."

        slots.add(slot)
        return f"Saved {hand}/{letter}/{destination.name}."

    def delete_last() -> str:
        if not slots:
            return f"{hand}/{letter} has no images to delete."

        slot = max(slots)
        target = image_path(output_dir, hand, letter, slot)
        try:
            target.unlink()
        except FileNotFoundError:
            slots.discard(slot)
            return f"{hand}/{letter}/{target.name} was already gone."
        except OSError as error:
            return f"Could not delete {target.name}: {error}"

        slots.discard(slot)
        return f"Deleted {hand}/{letter}/{target.name}."

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Camera stopped returning frames.")
                return 1

            now = time.monotonic()
            burst_state = "off"
            if not bursting and now >= next_rescan_at:
                # Pick up images deleted in Finder while the window stays open.
                slots = used_slots(output_dir, hand, letter, max_images)
                next_rescan_at = now + 1.0
            if bursting:
                if now < burst_starts_at:
                    burst_state = f"starts in {burst_starts_at - now:.1f}s"
                else:
                    burst_state = f"on ({args.burst_fps:g}/s)"
                    if now >= next_burst_at:
                        message = save_next(frame)
                        next_burst_at = now + burst_interval
                        if len(slots) >= max_images:
                            bursting = False
                            message = f"{hand}/{letter} reached {max_images} images."

            display = draw_status(
                frame,
                camera_label,
                hand,
                letter,
                len(slots),
                max_images,
                burst_state,
                message,
            )
            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # Escape
                break
            if key == ord("[") or key == ord("]"):
                step = -1 if key == ord("[") else 1
                selected_index = (selected_index + step) % len(LETTERS)
                letter = LETTERS[selected_index]
                slots = used_slots(output_dir, hand, letter, max_images)
                bursting = False
                message = f"Selected {hand}/{letter}."
                continue
            if key == ord(" "):
                bursting = False
                message = save_next(frame)
                continue
            if key in (8, 127):  # Backspace on Linux/Windows, Delete on macOS
                bursting = False
                message = delete_last()
                continue
            if key == ord("."):
                bursting = not bursting
                if bursting:
                    slots = used_slots(output_dir, hand, letter, max_images)
                    burst_starts_at = now + args.burst_delay
                    next_burst_at = burst_starts_at
                    message = f"Burst capture starting for {hand}/{letter}."
                else:
                    message = "Burst capture stopped."
                continue
            if key == ord("/"):
                hand = HANDS[1 - HANDS.index(hand)]
                slots = used_slots(output_dir, hand, letter, max_images)
                bursting = False
                message = f"Switched to {hand} hand."
                continue

            if key != 255:
                pressed = chr(key).upper()
                if pressed in LETTERS:
                    selected_index = LETTERS.index(pressed)
                    letter = pressed
                    slots = used_slots(output_dir, hand, letter, max_images)
                    bursting = False
                    message = f"Selected {hand}/{letter}."
    finally:
        camera.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
