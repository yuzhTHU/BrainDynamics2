#!/usr/bin/env python3
"""Recover candidate neural-frame timestamps from a flashing light in the face video."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def update_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--video", type=Path, default=Path("data/raw/51.mp4"))
    parser.add_argument("--output-dir", type=Path, default=Path("logs/recover_video_sync/initial"))
    parser.add_argument("--expected-neural-frames", type=int, default=28801)
    parser.add_argument("--video-fps", type=float, default=30.0)
    parser.add_argument("--ffmpeg", type=Path, default=Path("venv/bin/ffmpeg"))


def decode_light(video: Path, ffmpeg: Path) -> np.ndarray:
    command = [str(ffmpeg), "-v", "error", "-i", str(video), "-vf", "crop=100:60:280:10,scale=1:1:flags=area,format=rgb24", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(command, check=True, capture_output=True).stdout
    rgb = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(float)
    return rgb[:, 1] - 0.5 * (rgb[:, 0] + rgb[:, 2])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    update_parser(parser)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    signal = decode_light(args.video, args.ffmpeg)
    threshold = float((np.quantile(signal, 0.1) + np.quantile(signal, 0.9)) / 2)
    high = signal > threshold
    rising = np.flatnonzero(high[1:] & ~high[:-1]) + 1
    falling = np.flatnonzero(~high[1:] & high[:-1]) + 1

    gaps = np.diff(rising) / args.video_fps
    split = np.flatnonzero(gaps > 1.0) + 1
    runs = np.split(rising, split)
    candidate = max(runs, key=len)
    pulse_times = candidate / args.video_fps
    falling_position = np.searchsorted(falling, candidate, side="right")
    valid_width = falling_position < len(falling)
    pulse_width_frames = np.full(len(candidate), np.nan)
    pulse_width_frames[valid_width] = falling[falling_position[valid_width]] - candidate[valid_width]
    pulse_width_seconds = pulse_width_frames / args.video_fps
    pulse_index = np.arange(len(pulse_times))
    slope, intercept = np.polyfit(pulse_index, pulse_times, 1)
    fitted = intercept + slope * pulse_index
    residual = pulse_times - fitted
    pulse_table = pd.DataFrame({"candidate_neural_frame": pulse_index, "video_frame": candidate, "video_seconds": pulse_times, "pulse_width_video_frames": pulse_width_frames, "pulse_width_seconds": pulse_width_seconds, "linear_fit_seconds": fitted, "residual_seconds": residual})
    pulse_table.to_csv(args.output_dir / "candidate_frame_timestamps.csv", index=False)

    summary = {
        "threshold_green_excess": threshold,
        "total_video_frames_decoded": len(signal),
        "all_rising_edges": len(rising),
        "all_falling_edges": len(falling),
        "longest_pulse_run_count": len(candidate),
        "expected_neural_frames": args.expected_neural_frames,
        "frame_count_difference": len(candidate) - args.expected_neural_frames,
        "first_pulse_video_seconds": float(pulse_times[0]),
        "last_pulse_video_seconds": float(pulse_times[-1]),
        "pulse_run_duration_seconds": float(pulse_times[-1] - pulse_times[0]),
        "linear_clock_intercept_seconds": float(intercept),
        "linear_seconds_per_neural_frame": float(slope),
        "linear_sampling_hz": float(1.0 / slope),
        "timing_residual_std_seconds": float(np.std(residual)),
        "timing_residual_max_absolute_seconds": float(np.max(np.abs(residual))),
        "median_interpulse_seconds": float(np.median(np.diff(pulse_times))),
        "unique_interpulse_video_frames": np.unique(np.diff(candidate)).astype(int).tolist(),
        "pulse_width_video_frames_median": float(np.nanmedian(pulse_width_frames)),
        "pulse_width_video_frames_range": [float(np.nanmin(pulse_width_frames)), float(np.nanmax(pulse_width_frames))],
        "pulse_width_seconds_median": float(np.nanmedian(pulse_width_seconds)),
        "pulse_width_seconds_p10_p90": [float(np.nanquantile(pulse_width_seconds, 0.1)), float(np.nanquantile(pulse_width_seconds, 0.9))],
        "interpretation": "Candidate only: correspondence is strong if the longest run has 28,801 pulses, but hardware documentation would still be needed for formal confirmation.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
