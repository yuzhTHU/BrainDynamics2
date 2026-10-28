#!/usr/bin/env python3
"""Test event propagation in spatial bins built from cell coordinates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import detrend, find_peaks
from sklearn.cluster import KMeans


STATES = ("Awake", "NREM")


def update_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("logs/spatial_bin_event_propagation/initial"))
    parser.add_argument("--n-bins", type=int, default=36)
    parser.add_argument("--cells-per-bin", type=int, default=128)
    parser.add_argument("--sampling-hz", type=float, default=4.146313525439001)
    parser.add_argument("--window-seconds", type=float, default=3.0)
    parser.add_argument("--lag-search-seconds", type=float, default=1.2)
    parser.add_argument("--minimum-distance-seconds", type=float, default=3.0)
    parser.add_argument("--threshold-quantile", type=float, default=0.9)
    parser.add_argument("--polarity", choices=("positive", "negative"), default="positive")
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261009)


def make_segments(labels: np.ndarray) -> pd.DataFrame:
    changes = np.r_[True, labels[1:] != labels[:-1], True]
    boundaries = np.flatnonzero(changes)
    return pd.DataFrame([{"start": int(a), "stop": int(b), "state": str(labels[a])} for a, b in zip(boundaries[:-1], boundaries[1:], strict=True)])


def spatial_bins(node_info: pd.DataFrame, n_bins: int, cells_per_bin: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coordinates = node_info[["coords_2d_x", "coords_2d_y"]].to_numpy(dtype=np.float64)
    normalized = (coordinates - coordinates.mean(axis=0)) / coordinates.std(axis=0)
    labels = KMeans(n_clusters=n_bins, random_state=int(rng.integers(2**31 - 1)), n_init=20).fit_predict(normalized)
    selected, selected_bins, centers, total_counts = [], [], [], []
    for bin_index in range(n_bins):
        members = np.flatnonzero(labels == bin_index)
        choice = rng.choice(members, size=min(cells_per_bin, len(members)), replace=False)
        selected.extend(choice)
        selected_bins.extend([bin_index] * len(choice))
        centers.append(coordinates[members].mean(axis=0))
        total_counts.append(len(members))
    return np.asarray(selected), np.asarray(selected_bins), np.asarray(centers), np.asarray(total_counts)


def aggregate_bins(data: np.ndarray, selected: np.ndarray, selected_bins: np.ndarray, n_bins: int) -> np.ndarray:
    sampled = np.asarray(data[:, selected], dtype=np.float32)
    output = np.empty((sampled.shape[0], n_bins), dtype=np.float32)
    for bin_index in range(n_bins):
        output[:, bin_index] = sampled[:, selected_bins == bin_index].mean(axis=1)
    return output


def episode_detrend(values: np.ndarray, segments: pd.DataFrame) -> np.ndarray:
    output = np.empty_like(values, dtype=np.float64)
    for row in segments.itertuples(index=False):
        output[row.start : row.stop] = detrend(values[row.start : row.stop], axis=0, type="linear")
    return output


def detect_events(score: np.ndarray, table: pd.DataFrame, half_window: int, distance: int, quantile: float) -> tuple[np.ndarray, np.ndarray]:
    state_index = np.concatenate([np.arange(row.start, row.stop) for row in table.itertuples(index=False)])
    threshold = float(np.quantile(score[state_index], quantile))
    events, episodes = [], []
    for episode_id, row in enumerate(table.itertuples(index=False)):
        peaks, _ = find_peaks(score[row.start : row.stop], height=threshold, distance=distance)
        peaks = peaks[(peaks >= half_window) & (peaks < row.stop - row.start - half_window)]
        events.extend(row.start + peaks)
        episodes.extend([episode_id] * len(peaks))
    return np.asarray(events, dtype=int), np.asarray(episodes, dtype=int)


def fit_lag_plane(lags: np.ndarray, centers: np.ndarray) -> tuple[float, float, float]:
    spatial = (centers - centers.mean(axis=0)) / centers.std(axis=0)
    design = np.column_stack([np.ones(len(spatial)), spatial])
    coefficient = np.linalg.lstsq(design, lags, rcond=None)[0]
    prediction = design @ coefficient
    denominator = np.sum((lags - lags.mean()) ** 2)
    r2 = 1.0 - np.sum((lags - prediction) ** 2) / denominator if denominator > 0 else 0.0
    magnitude = float(np.linalg.norm(coefficient[1:]))
    angle = float(np.arctan2(coefficient[2], coefficient[1])) if magnitude > 0 else np.nan
    return float(r2), magnitude, angle


def analyze_state(z: np.ndarray, state: str, segments: pd.DataFrame, centers: np.ndarray, args: argparse.Namespace, rng: np.random.Generator) -> tuple[dict, pd.DataFrame, dict]:
    table = segments[segments.state == state].reset_index(drop=True)
    state_index = np.concatenate([np.arange(row.start, row.stop) for row in table.itertuples(index=False)])
    mean, std = z[state_index].mean(axis=0), z[state_index].std(axis=0)
    state_z = (z - mean) / np.where(std > 0, std, 1.0)
    score = state_z.mean(axis=1)
    detection_score = score if args.polarity == "positive" else -score
    half_window = int(round(args.window_seconds * args.sampling_hz))
    search = int(round(args.lag_search_seconds * args.sampling_hz))
    distance = int(round(args.minimum_distance_seconds * args.sampling_hz))
    events, episode_ids = detect_events(detection_score, table, half_window, distance, args.threshold_quantile)
    offsets = np.arange(-half_window, half_window + 1)
    tensor = np.stack([state_z[event + offsets] for event in events])
    center = half_window
    search_slice = slice(center - search, center + search + 1)
    timing_tensor = tensor if args.polarity == "positive" else -tensor
    event_lags = np.argmax(timing_tensor[:, search_slice, :], axis=1) - search
    event_lags_seconds = event_lags / args.sampling_hz
    plane_rows = []
    for event_index, lags in enumerate(event_lags_seconds):
        r2, magnitude, angle = fit_lag_plane(lags, centers)
        plane_rows.append({"state": state, "event": int(events[event_index]), "episode": int(episode_ids[event_index]), "plane_r2": r2, "gradient_seconds_per_spatial_sd": magnitude, "angle_radians": angle})
    planes = pd.DataFrame(plane_rows)
    null_medians, null_resultants = [], []
    for _ in range(args.permutations):
        r2_values, angles = [], []
        for lags in event_lags_seconds:
            r2, _, angle = fit_lag_plane(rng.permutation(lags), centers)
            r2_values.append(r2)
            angles.append(angle)
        finite = np.asarray(angles)[np.isfinite(angles)]
        null_medians.append(np.median(r2_values))
        null_resultants.append(np.abs(np.mean(np.exp(1j * finite))) if len(finite) else 0.0)
    finite_angles = planes.angle_radians[np.isfinite(planes.angle_radians)].to_numpy()
    resultant = float(np.abs(np.mean(np.exp(1j * finite_angles)))) if len(finite_angles) else 0.0
    observed_median_r2 = float(planes.plane_r2.median())
    template = tensor.mean(axis=0)
    timing_template = template if args.polarity == "positive" else -template
    template_lags = (np.argmax(timing_template[search_slice], axis=0) - search) / args.sampling_hz
    summary = {"state": state, "polarity": args.polarity, "n_events": int(len(events)), "n_episodes_with_events": int(len(np.unique(episode_ids))), "median_event_plane_r2": observed_median_r2, "plane_r2_permutation_p": float((1 + np.sum(np.asarray(null_medians) >= observed_median_r2)) / (args.permutations + 1)), "direction_resultant_length": resultant, "direction_permutation_p": float((1 + np.sum(np.asarray(null_resultants) >= resultant)) / (args.permutations + 1)), "template_bins_with_nonzero_peak_frame": int(np.sum(template_lags != 0))}
    arrays = {"template": template, "offsets_seconds": offsets / args.sampling_hz, "events": events, "episode_ids": episode_ids, "event_lags_seconds": event_lags_seconds, "template_lags_seconds": template_lags}
    return summary, planes, arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    update_parser(parser)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    node_info = pd.read_csv(args.data_dir / "node_info.csv", index_col=0)
    selected, selected_bins, centers, total_counts = spatial_bins(node_info, args.n_bins, args.cells_per_bin, rng)
    data = np.load(args.data_dir / "data.npy", mmap_mode="r")
    binned = aggregate_bins(data, selected, selected_bins, args.n_bins)
    labels = pd.read_csv(args.data_dir / "time_info.csv", index_col=0)["stimuli"].astype(str).to_numpy()
    segments = make_segments(labels)
    residual = episode_detrend(binned, segments)
    summaries, plane_tables = [], []
    save_arrays = {"centers": centers, "selected_nodes": selected, "selected_bins": selected_bins, "total_bin_counts": total_counts, "binned_raw": binned, "binned_detrended": residual}
    for state in STATES:
        summary, planes, arrays = analyze_state(residual, state, segments, centers, args, rng)
        summaries.append(summary)
        plane_tables.append(planes)
        save_arrays.update({f"{state}_{name}": value for name, value in arrays.items()})
    pd.DataFrame(summaries).to_csv(args.output_dir / "summary.csv", index=False)
    pd.concat(plane_tables, ignore_index=True).to_csv(args.output_dir / "event_plane_fits.csv", index=False)
    pd.DataFrame({"bin": np.arange(args.n_bins), "center_x": centers[:, 0], "center_y": centers[:, 1], "total_cells": total_counts, "sampled_cells": [np.sum(selected_bins == index) for index in range(args.n_bins)]}).to_csv(args.output_dir / "spatial_bins.csv", index=False)
    np.savez_compressed(args.output_dir / "event_arrays.npz", **save_arrays)
    (args.output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
