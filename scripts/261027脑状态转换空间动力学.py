#!/usr/bin/env python3
"""Estimate spatial recruitment order around EEG-defined state transitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from sklearn.isotonic import IsotonicRegression


TRANSITIONS = (("Awake", "NREM"), ("NREM", "Awake"))


def update_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--event-arrays", type=Path, default=Path("logs/spatial_bin_event_propagation/initial/event_arrays.npz"))
    parser.add_argument("--time-info", type=Path, default=Path("data/processed/time_info.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("logs/spatial_state_transitions/initial"))
    parser.add_argument("--sampling-hz", type=float, default=4.146313525439001)
    parser.add_argument("--window-seconds", type=float, default=25.0)
    parser.add_argument("--baseline-inner-seconds", type=float, default=5.0)
    parser.add_argument("--baseline-outer-seconds", type=float, default=20.0)
    parser.add_argument("--search-before-seconds", type=float, default=10.0)
    parser.add_argument("--search-after-seconds", type=float, default=15.0)
    parser.add_argument("--smooth-seconds", type=float, default=1.0)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261009)


def normalized_template(tensor: np.ndarray, offsets: np.ndarray, inner: float, outer: float, smooth_frames: int) -> np.ndarray:
    template = tensor.mean(axis=0)
    before = template[(offsets >= -outer) & (offsets <= -inner)].mean(axis=0)
    after = template[(offsets >= inner) & (offsets <= outer)].mean(axis=0)
    scale = after - before
    progress = (template - before) / np.where(np.abs(scale) > 1e-8, scale, 1.0)
    return uniform_filter1d(progress, size=smooth_frames, axis=0, mode="nearest")


def crossing_times(progress: np.ndarray, offsets: np.ndarray, before: float, after: float) -> np.ndarray:
    mask = (offsets >= -before) & (offsets <= after)
    selected = progress[mask]
    selected_offsets = offsets[mask]
    crossing = np.empty(selected.shape[1])
    for bin_index in range(selected.shape[1]):
        fitted = IsotonicRegression(increasing=True, out_of_bounds="clip").fit_transform(selected_offsets, selected[:, bin_index])
        above = np.flatnonzero(fitted >= .5)
        if not len(above):
            crossing[bin_index] = selected_offsets[-1]
        elif above[0] == 0:
            crossing[bin_index] = selected_offsets[0]
        else:
            right = above[0]
            left = right - 1
            weight = (.5 - fitted[left]) / max(fitted[right] - fitted[left], 1e-12)
            crossing[bin_index] = selected_offsets[left] + weight * (selected_offsets[right] - selected_offsets[left])
    return crossing


def fit_plane(lags: np.ndarray, centers: np.ndarray) -> tuple[float, float, float]:
    spatial = (centers - centers.mean(axis=0)) / centers.std(axis=0)
    design = np.column_stack([np.ones(len(spatial)), spatial])
    coefficient = np.linalg.lstsq(design, lags, rcond=None)[0]
    prediction = design @ coefficient
    denominator = np.sum((lags - lags.mean()) ** 2)
    r2 = 1.0 - np.sum((lags - prediction) ** 2) / denominator if denominator > 0 else 0.0
    angle = float(np.arctan2(coefficient[2], coefficient[1]))
    return float(r2), float(np.linalg.norm(coefficient[1:])), angle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    update_parser(parser)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    archive = np.load(args.event_arrays)
    values, centers = archive["binned_raw"].astype(np.float64), archive["centers"].astype(np.float64)
    labels = pd.read_csv(args.time_info, index_col=0)["stimuli"].astype(str).to_numpy()
    half_window = int(round(args.window_seconds * args.sampling_hz))
    offsets = np.arange(-half_window, half_window + 1) / args.sampling_hz
    smooth_frames = max(1, int(round(args.smooth_seconds * args.sampling_hz)))
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    summaries, lag_rows = [], []
    save = {"centers": centers, "offsets_seconds": offsets}
    for before_state, after_state in TRANSITIONS:
        transition_name = f"{before_state}_to_{after_state}"
        events = np.asarray([index for index in change if labels[index - 1] == before_state and labels[index] == after_state and index >= half_window and index + half_window < len(values)], dtype=int)
        tensor = np.stack([values[index - half_window : index + half_window + 1] for index in events])
        progress = normalized_template(tensor, offsets, args.baseline_inner_seconds, args.baseline_outer_seconds, smooth_frames)
        lags = crossing_times(progress, offsets, args.search_before_seconds, args.search_after_seconds)
        r2, magnitude, angle = fit_plane(lags, centers)
        bootstrap_lags, bootstrap_angles, bootstrap_r2 = [], [], []
        for _ in range(args.bootstrap_replicates):
            sample = rng.integers(0, len(events), size=len(events))
            candidate = normalized_template(tensor[sample], offsets, args.baseline_inner_seconds, args.baseline_outer_seconds, smooth_frames)
            candidate_lags = crossing_times(candidate, offsets, args.search_before_seconds, args.search_after_seconds)
            candidate_r2, _, candidate_angle = fit_plane(candidate_lags, centers)
            bootstrap_lags.append(candidate_lags)
            bootstrap_angles.append(candidate_angle)
            bootstrap_r2.append(candidate_r2)
        null_r2 = [fit_plane(rng.permutation(lags), centers)[0] for _ in range(args.permutations)]
        bootstrap_lags = np.asarray(bootstrap_lags)
        resultant = float(np.abs(np.mean(np.exp(1j * np.asarray(bootstrap_angles)))))
        summaries.append({"transition": transition_name, "n_events": len(events), "plane_r2": r2, "plane_permutation_p": float((1 + np.sum(np.asarray(null_r2) >= r2)) / (args.permutations + 1)), "gradient_seconds_per_spatial_sd": magnitude, "angle_degrees": float(np.degrees(angle)), "bootstrap_angle_resultant": resultant, "bootstrap_plane_r2_median": float(np.median(bootstrap_r2))})
        for bin_index, lag in enumerate(lags):
            lag_rows.append({"transition": transition_name, "bin": bin_index, "crossing_seconds": lag, "bootstrap_median_seconds": float(np.median(bootstrap_lags[:, bin_index])), "bootstrap_q025_seconds": float(np.quantile(bootstrap_lags[:, bin_index], .025)), "bootstrap_q975_seconds": float(np.quantile(bootstrap_lags[:, bin_index], .975))})
        save.update({f"{transition_name}_events": events, f"{transition_name}_progress": progress, f"{transition_name}_lags": lags, f"{transition_name}_bootstrap_lags": bootstrap_lags})
    pd.DataFrame(summaries).to_csv(args.output_dir / "summary.csv", index=False)
    pd.DataFrame(lag_rows).to_csv(args.output_dir / "bin_crossing_times.csv", index=False)
    np.savez_compressed(args.output_dir / "transition_arrays.npz", **save)
    (args.output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
