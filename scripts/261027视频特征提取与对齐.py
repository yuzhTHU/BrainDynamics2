#!/usr/bin/env python3
"""Extract face-video covariates and propose an alignment to neural state labels."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import correlate


def update_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--video", type=Path, default=Path("data/raw/51.mp4"))
    parser.add_argument("--time-info", type=Path, default=Path("data/processed/time_info.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("logs/extract_align_video/initial"))
    parser.add_argument("--sampling-hz", type=float, default=4.146313525439001)
    parser.add_argument("--sync-summary", type=Path, default=Path("logs/recover_video_sync/initial/summary.json"))
    parser.add_argument("--sync-timestamps", type=Path, default=Path("logs/recover_video_sync/initial/candidate_frame_timestamps.csv"))
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--height", type=int, default=72)
    parser.add_argument("--ffmpeg", type=Path, default=Path("venv/bin/ffmpeg"))


def extract_features(args: argparse.Namespace) -> pd.DataFrame:
    filter_spec = f"fps={args.sampling_hz},scale={args.width}:{args.height}:flags=area,format=rgb24"
    command = [str(args.ffmpeg), "-v", "error", "-i", str(args.video), "-vf", filter_spec, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    frame_bytes = args.width * args.height * 3
    previous_gray = None
    rows = []
    index = 0
    while True:
        raw = process.stdout.read(frame_bytes)
        if not raw:
            break
        if len(raw) != frame_bytes:
            raise RuntimeError(f"Incomplete decoded frame: {len(raw)} of {frame_bytes} bytes")
        rgb = np.frombuffer(raw, dtype=np.uint8).reshape(args.height, args.width, 3).astype(np.float32)
        gray = rgb.mean(axis=2)
        face = gray[10:68, 0:72]
        eye = gray[25:48, 43:65]
        green = rgb[2:14, 43:58]
        if previous_gray is None:
            global_motion = np.nan
            face_motion = np.nan
        else:
            difference = np.abs(gray - previous_gray)
            global_motion = float(difference.mean())
            face_motion = float(difference[10:68, 0:72].mean())
        rows.append({"video_frame_sampled": index, "video_seconds": index / args.sampling_hz, "global_motion": global_motion, "face_motion": face_motion, "eye_mean": float(eye.mean()), "eye_std": float(eye.std()), "eye_q10": float(np.quantile(eye, 0.1)), "eye_dark_fraction": float(np.mean(eye < 80.0)), "green_excess": float(green[:, :, 1].mean() - 0.5 * (green[:, :, 0].mean() + green[:, :, 2].mean()))})
        previous_gray = gray
        index += 1
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"ffmpeg exited with status {return_code}")
    return pd.DataFrame(rows)


def rolling_pearson(video: np.ndarray, neural: np.ndarray) -> np.ndarray:
    video = np.nan_to_num(video, nan=np.nanmedian(video))
    neural = neural.astype(np.float64)
    n = len(neural)
    cross = correlate(video, neural, mode="valid", method="fft")
    cumulative = np.r_[0.0, np.cumsum(video)]
    cumulative_sq = np.r_[0.0, np.cumsum(video * video)]
    window_sum = cumulative[n:] - cumulative[:-n]
    window_sq = cumulative_sq[n:] - cumulative_sq[:-n]
    neural_sum = neural.sum()
    neural_centered_sq = np.sum((neural - neural.mean()) ** 2)
    covariance = cross - window_sum * neural_sum / n
    video_centered_sq = window_sq - window_sum**2 / n
    denominator = np.sqrt(np.maximum(video_centered_sq * neural_centered_sq, 0.0))
    return np.divide(covariance, denominator, out=np.full_like(covariance, np.nan), where=denominator > 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    update_parser(parser)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features = extract_features(args)
    features.to_csv(args.output_dir / "video_features_4hz.csv", index=False)

    labels = pd.read_csv(args.time_info, index_col=0)["stimuli"].astype(str).to_numpy()
    awake = (labels == "Awake").astype(float)
    scans = pd.DataFrame({"offset_frames": np.arange(len(features) - len(labels) + 1)})
    scans["offset_seconds"] = scans.offset_frames / args.sampling_hz
    for name in ("global_motion", "face_motion"):
        scans[f"pearson_awake_{name}"] = rolling_pearson(features[name].to_numpy(), awake)
    scans.to_csv(args.output_dir / "alignment_scan.csv", index=False)

    score_columns = [column for column in scans if column.startswith("pearson_awake_")]
    best = {}
    for column in score_columns:
        index = int(scans[column].idxmax())
        best[column] = {"offset_frames": int(scans.loc[index, "offset_frames"]), "offset_seconds": float(scans.loc[index, "offset_seconds"]), "pearson": float(scans.loc[index, column])}
    sync = json.loads(args.sync_summary.read_text(encoding="utf-8"))
    pulse_times = pd.read_csv(args.sync_timestamps)
    if len(pulse_times) != len(labels):
        raise ValueError(f"Pulse count {len(pulse_times)} does not match neural frame count {len(labels)}")
    feature_times = features["video_seconds"].to_numpy()
    targets = pulse_times["video_seconds"].to_numpy()
    numeric = ["global_motion", "face_motion", "eye_mean", "eye_std", "eye_q10", "eye_dark_fraction", "green_excess"]
    aligned = pd.DataFrame({column: np.interp(targets, feature_times, features[column].bfill().to_numpy()) for column in numeric})
    aligned["neural_frame"] = np.arange(len(labels))
    aligned["pulse_video_seconds"] = targets
    aligned["state"] = labels
    aligned.to_csv(args.output_dir / "aligned_video_features.csv", index=False)
    state_summary = aligned.groupby("state")[numeric].agg(["mean", "median", "std"])
    state_summary.columns = [f"{feature}_{statistic}" for feature, statistic in state_summary.columns]
    state_summary.reset_index().to_csv(args.output_dir / "aligned_state_video_summary.csv", index=False)
    summary = {"video_samples": len(features), "neural_samples": len(labels), "sampling_hz": args.sampling_hz, "valid_offset_range_seconds": [0.0, float((len(features) - len(labels)) / args.sampling_hz)], "best_motion_alignments": best, "sync_intercept_seconds": sync["linear_clock_intercept_seconds"], "alignment_method": "linear interpolation of uniformly sampled video features at each detected optical pulse timestamp", "warning": "The alignment uses the 28,801-pulse optical synchronization candidate. Behavior-only alignment is retained only as a weak diagnostic."}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
