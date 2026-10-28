#!/usr/bin/env python3
"""Separate apparent neural memory into episode offsets, trends, and faster residuals."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import detrend


STATES = ("Awake", "NREM", "REM")


def update_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--audit-dir", type=Path, default=Path("logs/audit_data/initial"))
    parser.add_argument("--output-dir", type=Path, default=Path("logs/timescale_decomposition/initial"))
    parser.add_argument("--sampling-hz", type=float, default=4.146313525439001)
    parser.add_argument("--lags", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    parser.add_argument("--highpass-seconds", type=float, nargs="+", default=[16.0, 64.0])


def make_segments(labels: np.ndarray) -> pd.DataFrame:
    changes = np.r_[True, labels[1:] != labels[:-1], True]
    boundaries = np.flatnonzero(changes)
    return pd.DataFrame([{"start": int(a), "stop": int(b), "state": str(labels[a])} for a, b in zip(boundaries[:-1], boundaries[1:], strict=True)])


def transforms(block: np.ndarray, sampling_hz: float, highpass_seconds: list[float]):
    yield "raw", block
    yield "episode_centered", block - block.mean(axis=0, keepdims=True)
    if len(block) >= 3:
        yield "linear_detrended", detrend(block, axis=0, type="linear").astype(np.float32, copy=False)
    else:
        yield "linear_detrended", block - block.mean(axis=0, keepdims=True)
    for seconds in highpass_seconds:
        width = max(3, int(round(seconds * sampling_hz)))
        width += 1 - width % 2
        baseline = uniform_filter1d(block, size=width, axis=0, mode="reflect")
        yield f"highpass_{seconds:g}s", block - baseline


def empty_stats(n_nodes: int) -> dict[str, np.ndarray | int]:
    return {"n": 0, "sx": np.zeros(n_nodes), "sy": np.zeros(n_nodes), "sxx": np.zeros(n_nodes), "syy": np.zeros(n_nodes), "sxy": np.zeros(n_nodes)}


def update_stats(stats: dict[str, np.ndarray | int], left: np.ndarray, right: np.ndarray) -> None:
    stats["n"] += len(left)
    stats["sx"] += left.sum(axis=0, dtype=np.float64)
    stats["sy"] += right.sum(axis=0, dtype=np.float64)
    stats["sxx"] += np.einsum("ij,ij->j", left, left, dtype=np.float64)
    stats["syy"] += np.einsum("ij,ij->j", right, right, dtype=np.float64)
    stats["sxy"] += np.einsum("ij,ij->j", left, right, dtype=np.float64)


def correlations(stats: dict[str, np.ndarray | int]) -> np.ndarray:
    n = int(stats["n"])
    covariance = stats["sxy"] - stats["sx"] * stats["sy"] / n
    var_x = stats["sxx"] - stats["sx"] ** 2 / n
    var_y = stats["syy"] - stats["sy"] ** 2 / n
    denominator = np.sqrt(np.maximum(var_x * var_y, 0))
    return np.divide(covariance, denominator, out=np.full_like(covariance, np.nan), where=denominator > 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    update_parser(parser)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    indices = np.load(args.audit_dir / "sample_node_index.npy")
    data = np.load(args.data_dir / "data.npy", mmap_mode="r")
    labels = pd.read_csv(args.data_dir / "time_info.csv", index_col=0)["stimuli"].astype(str).to_numpy()
    segments = make_segments(labels)
    sample = np.asarray(data[:, indices], dtype=np.float32)
    mean = sample.mean(axis=0, dtype=np.float64)
    std = sample.std(axis=0, dtype=np.float64)
    z = ((sample - mean) / np.where(std > 0, std, 1.0)).astype(np.float32)

    method_names = ["raw", "episode_centered", "linear_detrended", *[f"highpass_{seconds:g}s" for seconds in args.highpass_seconds]]
    accumulators = {(state, method, lag): empty_stats(z.shape[1]) for state in STATES for method in method_names for lag in args.lags}
    variance_rows = []
    for segment in segments.itertuples(index=False):
        block = z[segment.start : segment.stop]
        raw_energy = float(np.mean(block.astype(np.float64) ** 2))
        for method, values in transforms(block, args.sampling_hz, args.highpass_seconds):
            energy = float(np.mean(values.astype(np.float64) ** 2))
            variance_rows.append({"state": segment.state, "start": segment.start, "stop": segment.stop, "method": method, "energy": energy, "raw_energy": raw_energy, "energy_fraction": energy / raw_energy if raw_energy > 0 else np.nan})
            for lag in args.lags:
                if len(values) <= lag:
                    continue
                update_stats(accumulators[(segment.state, method, lag)], values[:-lag], values[lag:])

    rows = []
    for (state, method, lag), stats in accumulators.items():
        if not stats["n"]:
            continue
        corr = correlations(stats)
        rows.append({"state": state, "method": method, "lag_steps": lag, "seconds": lag / args.sampling_hz, "mean": float(np.nanmean(corr)), "median": float(np.nanmedian(corr)), "p10": float(np.nanquantile(corr, 0.1)), "p90": float(np.nanquantile(corr, 0.9)), "n_pairs": int(stats["n"])})
    pd.DataFrame(rows).to_csv(args.output_dir / "autocorrelation_by_preprocessing.csv", index=False)
    pd.DataFrame(variance_rows).to_csv(args.output_dir / "episode_energy_fractions.csv", index=False)


if __name__ == "__main__":
    main()
