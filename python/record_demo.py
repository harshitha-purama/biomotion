"""
Hands-free demo recorder for the README / social clip.

No keypresses needed: shows a short countdown (to give you time to position
the camera so only your hand(s) are visible -- no face needed), then
auto-records for a fixed duration with the glowing trail (index tip, full
trail; thumb tip, connected pinch-line so both are visibly tracked) + a live
tremor-frequency readout per hand, then auto-stops and saves:
  - <out>.mp4, <out>.gif -- the clip
  - <out>.csv            -- full 21-landmark log (same schema as capture.py)
  - <out>_tremor_<hand>.png -- the actual PSD/frequency-peak plot for each
    hand recorded, generated the same way as analyze_tremor.py, so the demo
    video and its frequency graph are from the exact same session.

Usage:
  python record_demo.py --seconds 10 --out ../docs/demo
"""

import argparse
import csv
import time
from collections import deque
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import landmarks as LM
from load_csv import available_hands, load_hand_track
from tremor_math import dominant_freq, resample_uniform

MODEL_PATH = Path(__file__).parent / "models" / "hand_landmarker.task"
TRAIL_LEN = 40
HAND_COLORS = {"Left": (255, 200, 40), "Right": (60, 140, 255)}
FREQ_BUF_SECONDS = 2.5


def make_glow_layer(shape):
    return np.zeros(shape, dtype=np.uint8)


def draw_trail(glow, trail, color):
    for i in range(1, len(trail)):
        thickness = max(2, int(8 * (i / len(trail))))
        cv2.line(glow, trail[i - 1], trail[i], color, thickness, cv2.LINE_AA)
    if trail:
        cv2.circle(glow, trail[-1], 9, (255, 255, 255), -1, cv2.LINE_AA)


def draw_pinch(glow, index_pt, thumb_pt, color):
    cv2.line(glow, index_pt, thumb_pt, color, 2, cv2.LINE_AA)
    cv2.circle(glow, thumb_pt, 6, color, -1, cv2.LINE_AA)


def save_tremor_plot(csv_path, hand, out_png):
    t, xy, hand_used = load_hand_track(csv_path, LM.INDEX_TIP, hand=hand)
    if len(t) < 20:
        print(f"  [{hand}] not enough samples for a frequency plot ({len(t)} rows)")
        return None

    t_u, x_u, y_u, fs = resample_uniform(t, xy[:, 0], xy[:, 1])
    peak_freq, peak_power, f, power, x_d, y_d = dominant_freq(x_u, y_u, fs)

    fig, axes = plt.subplots(2, 1, figsize=(9, 6))
    axes[0].plot(t_u, x_d, label="x (detrended, high-passed)")
    axes[0].plot(t_u, y_d, label="y (detrended, high-passed)", alpha=0.7)
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("normalized displacement")
    axes[0].set_title(f"Fingertip trajectory -- {hand_used} hand")
    axes[0].legend()

    axes[1].plot(f, power)
    axes[1].axvspan(2.0, 15.0, color="orange", alpha=0.15, label="tremor band")
    if peak_freq is not None:
        axes[1].axvline(peak_freq, color="red", linestyle="--",
                         label=f"peak {peak_freq:.2f} Hz")
    axes[1].set_xlim(0, 20)
    axes[1].set_xlabel("frequency (Hz)")
    axes[1].set_ylabel("power")
    axes[1].set_title("Power spectral density (Welch)")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    if peak_freq is not None:
        print(f"  [{hand_used}] peak frequency: {peak_freq:.2f} Hz -> {out_png}")
    else:
        print(f"  [{hand_used}] no clear peak found -> {out_png}")
    return peak_freq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--countdown", type=float, default=3.0)
    parser.add_argument("--out", type=str, default="../docs/demo")
    parser.add_argument("--num-hands", type=int, default=2)
    parser.add_argument("--gif", action="store_true", default=True)
    parser.add_argument("--no-gif", dest="gif", action="store_false")
    parser.add_argument("--gif-fps", type=int, default=8)
    parser.add_argument("--gif-max-width", type=int, default=360)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file missing: {MODEL_PATH}. See README setup instructions.")

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

    ok, probe = cap.read()
    if not ok:
        raise RuntimeError("Could not read a frame from the webcam.")
    h, w = probe.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    mp4_path = out_path.with_suffix(".mp4")
    target_fps = 20.0
    writer = cv2.VideoWriter(str(mp4_path), fourcc, target_fps, (w, h))

    csv_path = out_path.with_suffix(".csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(LM.csv_columns())

    trails = {}
    glow = make_glow_layer((h, w, 3))
    freq_bufs = {}  # hand_label -> deque of (t_sec, x, y)

    window_name = "biomotion demo recorder (position hands only, no keypress needed)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    stream_start_ms = time.time() * 1000
    last_ts_ms = -1
    gif_frames = []
    gif_stride = max(1, round(target_fps / args.gif_fps))

    def detect(frame_bgr):
        nonlocal last_ts_ms
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(time.time() * 1000 - stream_start_ms)
        if ts_ms <= last_ts_ms:
            ts_ms = last_ts_ms + 1
        last_ts_ms = ts_ms
        return landmarker.detect_for_video(mp_image, ts_ms)

    # --- countdown phase (not recorded) ---
    countdown_start = time.time()
    print(f"Position both hands in frame (no face needed). Recording starts in "
          f"{args.countdown:.0f}s...")
    while time.time() - countdown_start < args.countdown:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        remaining = args.countdown - (time.time() - countdown_start)
        cv2.putText(frame, f"starting in {remaining:.1f}s", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 255), 3, cv2.LINE_AA)
        cv2.imshow(window_name, frame)
        cv2.waitKey(1)

    # --- recording phase ---
    print(f"Recording for {args.seconds:.0f}s...")
    rec_start = time.time()
    frame_idx = 0
    frame_count = 0
    while time.time() - rec_start < args.seconds:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        result = detect(frame)

        glow[:] = (glow.astype(np.float32) * 0.85).astype(np.uint8)

        now_sec = time.time()
        elapsed = time.time() - rec_start
        freq_lines = []
        if result.hand_landmarks:
            for lm, handedness in zip(result.hand_landmarks, result.handedness):
                hand_label = handedness[0].category_name
                hand_score = handedness[0].score
                color = HAND_COLORS.get(hand_label, (200, 200, 200))

                index_tip = lm[LM.INDEX_TIP]
                thumb_tip = lm[LM.THUMB_TIP]
                ix, iy = int(index_tip.x * w), int(index_tip.y * h)
                tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)

                trails.setdefault(hand_label, deque(maxlen=TRAIL_LEN)).append((ix, iy))
                draw_trail(glow, trails[hand_label], color)
                draw_pinch(glow, (ix, iy), (tx, ty), color)

                csv_writer.writerow(
                    LM.landmarks_to_row(frame_idx, elapsed, hand_label, hand_score, lm)
                )

                buf = freq_bufs.setdefault(hand_label, deque())
                buf.append((now_sec, index_tip.x, index_tip.y))
                while buf and now_sec - buf[0][0] > FREQ_BUF_SECONDS:
                    buf.popleft()

                freq_text = f"{hand_label}: --"
                if len(buf) >= 20:
                    ts = np.array([p[0] for p in buf])
                    xs = np.array([p[1] for p in buf])
                    ys = np.array([p[2] for p in buf])
                    _, xu, yu, fs = resample_uniform(ts, xs, ys)
                    freq, _, *_ = dominant_freq(xu, yu, fs)
                    freq_text = f"{hand_label}: {freq:.2f} Hz" if freq else f"{hand_label}: (no peak)"
                freq_lines.append((freq_text, color))
            frame_idx += 1

        blended = cv2.add(frame, cv2.GaussianBlur(glow, (0, 0), 6))

        cv2.putText(blended, f"REC  {elapsed:0.1f}/{args.seconds:0.0f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        for j, (freq_text, color) in enumerate(freq_lines):
            cv2.putText(blended, freq_text, (10, 60 + j * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        writer.write(blended)
        if frame_count % gif_stride == 0:
            small = cv2.resize(
                blended, (args.gif_max_width, int(h * args.gif_max_width / w))
            )
            gif_frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        frame_count += 1

        cv2.imshow(window_name, blended)
        cv2.waitKey(1)

    writer.release()
    csv_file.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved video -> {mp4_path}")
    print(f"Saved landmark log -> {csv_path}")

    # Plots first and unconditionally -- CSV is already safely on disk, so a
    # crash in GIF export (below) can never prevent these from being generated.
    print("Generating frequency-peak plots from this session:")
    for hand in sorted(available_hands(csv_path)):
        out_png = out_path.parent / f"{out_path.stem}_tremor_{hand}.png"
        save_tremor_plot(csv_path, hand, out_png)

    if args.gif and gif_frames:
        try:
            import imageio
            gif_path = out_path.with_suffix(".gif")
            imageio.mimsave(gif_path, gif_frames, format="GIF", fps=args.gif_fps,
                             subrectangles=True)
            print(f"Saved gif -> {gif_path}")
        except Exception as e:
            print(f"GIF export failed ({e}); regenerate separately with: "
                  f"python make_gif.py {mp4_path} --start-sec {args.countdown:.0f}")


if __name__ == "__main__":
    main()
