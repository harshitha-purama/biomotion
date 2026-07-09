"""
Webcam hand-tracking capture tool.

- Tracks up to 2 hands x 21 landmarks/hand via MediaPipe HandLandmarker,
  with left/right handedness labeling -> enables bilateral (left vs right)
  comparison, which is where a lot of the clinical signal actually lives
  (e.g. Parkinson's tremor is often markedly asymmetric).
- Renders a glowing per-hand fingertip trail (cyan = Left, orange = Right).
- Logs the full landmark set per hand per frame to CSV (one row per hand)
  so any downstream metric -- tremor FFT, tap decrement, joint-angle ROM,
  bilateral asymmetry -- can be computed later without re-recording.

Controls:
  r - start / stop recording to CSV
  c - clear the on-screen trail
  q - quit
"""

import argparse
import csv
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import landmarks as LM

MODEL_PATH = Path(__file__).parent / "models" / "hand_landmarker.task"

TRAIL_LEN = 40
HAND_COLORS = {"Left": (255, 200, 40), "Right": (60, 140, 255)}  # BGR: cyan-ish, orange-ish


def make_glow_layer(shape):
    return np.zeros(shape, dtype=np.uint8)


def draw_trail(glow, trail, color):
    for i in range(1, len(trail)):
        thickness = max(2, int(8 * (i / len(trail))))
        cv2.line(glow, trail[i - 1], trail[i], color, thickness, cv2.LINE_AA)
    if trail:
        cv2.circle(glow, trail[-1], 9, (255, 255, 255), -1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--out", type=str, default="../data")
    parser.add_argument("--session", type=str, default=None,
                         help="Session name used for the output CSV filename")
    parser.add_argument("--num-hands", type=int, default=2)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file missing: {MODEL_PATH}. Download it with:\n"
            f'curl -L -o "{MODEL_PATH}" '
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task"
        )

    landmarker = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=args.num_hands,
            min_hand_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try a different --camera index.")

    trails = defaultdict(lambda: deque(maxlen=TRAIL_LEN))
    glow = None
    recording = False
    csv_file = None
    csv_writer = None
    t0 = None
    frame_idx = 0
    prev_time = time.time()
    stream_start_ms = time.time() * 1000
    last_ts_ms = -1

    print("Controls: [r] start/stop recording  [c] clear trail  [q] quit")
    print("IMPORTANT: click the video window titlebar to give it focus BEFORE "
          "pressing r/c/q -- keys typed into this terminal are not seen by OpenCV.")

    window_name = "biomotion capture (CLICK HERE, then press r/c/q)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        if glow is None:
            glow = make_glow_layer(frame.shape)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(time.time() * 1000 - stream_start_ms)
        if ts_ms <= last_ts_ms:
            ts_ms = last_ts_ms + 1
        last_ts_ms = ts_ms
        result = landmarker.detect_for_video(mp_image, ts_ms)

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        glow[:] = (glow.astype(np.float32) * 0.85).astype(np.uint8)  # fade every frame

        detected_labels = []
        if result.hand_landmarks:
            elapsed = now - t0 if recording else 0.0
            for lm, handedness in zip(result.hand_landmarks, result.handedness):
                # NOTE: frame is mirrored (selfie view) before detection, so this
                # label already corresponds to the user's own left/right hand.
                hand_label = handedness[0].category_name
                hand_score = handedness[0].score
                detected_labels.append(hand_label)
                color = HAND_COLORS.get(hand_label, (200, 200, 200))

                tip = lm[LM.INDEX_TIP]
                ix, iy = int(tip.x * w), int(tip.y * h)
                trails[hand_label].append((ix, iy))
                draw_trail(glow, trails[hand_label], color)

                thumb = lm[LM.THUMB_TIP]
                tx, ty = int(thumb.x * w), int(thumb.y * h)
                cv2.circle(frame, (tx, ty), 6, color, -1, cv2.LINE_AA)
                cv2.putText(frame, hand_label, (ix + 12, iy - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

                if recording and csv_writer:
                    csv_writer.writerow(
                        LM.landmarks_to_row(frame_idx, elapsed, hand_label, hand_score, lm)
                    )
            if recording:
                frame_idx += 1

        blended = cv2.add(frame, cv2.GaussianBlur(glow, (0, 0), 6))

        status = f"REC {frame_idx}f" if recording else "idle"
        hands_str = "+".join(detected_labels) if detected_labels else "no hand"
        cv2.putText(blended, f"{status}  fps:{fps:.0f}  [{hands_str}]", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255) if recording else (200, 200, 200), 2, cv2.LINE_AA)
        cv2.putText(blended, "click this window for r/c/q to work", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 2, cv2.LINE_AA)

        cv2.imshow(window_name, blended)
        raw_key = cv2.waitKey(1)
        key = chr(raw_key & 0xFF).lower() if raw_key != -1 else ""

        if key == "q":
            break
        elif key == "c":
            trails.clear()
        elif key == "r":
            recording = not recording
            if recording:
                name = args.session or time.strftime("%Y%m%d_%H%M%S")
                path = out_dir / f"{name}.csv"
                csv_file = open(path, "w", newline="")
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(LM.csv_columns())
                t0 = time.time()
                frame_idx = 0
                print(f"Recording -> {path}")
            else:
                if csv_file:
                    csv_file.close()
                    csv_file = None
                print("Stopped recording.")

    if csv_file:
        csv_file.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
