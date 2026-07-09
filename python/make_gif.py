"""
Convert a recorded .mp4 (e.g. from record_demo.py) into a .gif for the README.

Decoupled from the live capture loop on purpose -- if GIF export fails
inside record_demo.py (or you just want a different clip/crop/duration),
you can re-run this against the already-saved video without re-recording.

Reads frames via OpenCV (no ffmpeg binary needed) and writes the GIF via
imageio's Pillow backend (also no ffmpeg needed).
"""

import argparse
from pathlib import Path

import cv2
import imageio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=str)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--max-width", type=int, default=480)
    parser.add_argument("--start-sec", type=float, default=0.0,
                         help="Skip this many seconds from the start (e.g. to cut the countdown)")
    parser.add_argument("--duration-sec", type=float, default=None,
                         help="Only include this many seconds after --start-sec")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(args.start_sec * src_fps)
    end_frame = total_frames if args.duration_sec is None else min(
        total_frames, start_frame + int(args.duration_sec * src_fps)
    )
    stride = max(1, round(src_fps / args.fps))

    out_w = min(args.max_width, w)
    out_h = int(h * out_w / w)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    idx = start_frame
    while idx < end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        if (idx - start_frame) % stride == 0:
            small = cv2.resize(frame, (out_w, out_h))
            frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()

    if not frames:
        raise SystemExit("No frames read from video -- check --start-sec/--duration-sec range.")

    out_path = Path(args.out) if args.out else Path(args.video_path).with_suffix(".gif")
    imageio.mimsave(out_path, frames, format="GIF", fps=args.fps, subrectangles=True)
    size_mb = out_path.stat().st_size / 1e6
    print(f"Wrote {len(frames)} frames -> {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
