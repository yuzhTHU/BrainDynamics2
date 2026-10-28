#!/usr/bin/env python3
"""Audit the neural activity-trace dataset without loading it fully into RAM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.signal import welch
from sklearn.decomposition import PCA


STATES = ("Awake", "NREM", "REM")


def update_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("logs/audit_data/initial"))
    parser.add_argument("--sample-nodes", type=int, default=2048)
    parser.add_argument("--pca-nodes", type=int, default=1024)
    parser.add_argument("--pca-components", type=int, default=32)
    parser.add_argument("--chunk-rows", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--sampling-hz", type=float, default=4.146313525439001)


def contiguous_segments(labels: np.ndarray) -> pd.DataFrame:
    changes = np.r_[True, labels[1:] != labels[:-1], True]
    boundaries = np.flatnonzero(changes)
    rows = []
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        rows.append({"start": int(start), "stop": int(stop), "length": int(stop - start), "state": str(labels[start])})
    return pd.DataFrame(rows)


def sample_nodes_stratified(node_info: pd.DataFrame, n_nodes: int, rng: np.random.Generator) -> np.ndarray:
    n_nodes = min(n_nodes, len(node_info))
    counts = node_info["acs"].value_counts()
    allocation = np.maximum(1, np.floor(n_nodes * counts / counts.sum()).astype(int))
    while allocation.sum() > n_nodes:
        key = allocation[allocation > 1].idxmax()
        allocation[key] -= 1
    while allocation.sum() < n_nodes:
        key = (counts - allocation).idxmax()
        allocation[key] += 1
    selected = []
    for area, count in allocation.items():
        candidates = np.flatnonzero(node_info["acs"].to_numpy() == area)
        selected.extend(rng.choice(candidates, size=min(int(count), len(candidates)), replace=False).tolist())
    return np.sort(np.asarray(selected, dtype=np.int64))


def exact_scan_and_region_means(data: np.memmap, region_codes: np.ndarray, n_regions: int, chunk_rows: int) -> tuple[dict, np.ndarray]:
    n_time, n_node = data.shape
    membership = sparse.csr_matrix((np.ones(n_node, dtype=np.float32), (np.arange(n_node), region_codes)), shape=(n_node, n_regions))
    region_counts = np.bincount(region_codes, minlength=n_regions).astype(np.float64)
    region_means = np.empty((n_time, n_regions), dtype=np.float32)
    total = 0.0
    total_sq = 0.0
    count = 0
    minimum = np.inf
    maximum = -np.inf
    n_nan = 0
    n_inf = 0
    for start in range(0, n_time, chunk_rows):
        stop = min(start + chunk_rows, n_time)
        block = np.asarray(data[start:stop], dtype=np.float32)
        finite = np.isfinite(block)
        n_nan += int(np.isnan(block).sum())
        n_inf += int(np.isinf(block).sum())
        values = block[finite].astype(np.float64)
        if values.size:
            minimum = min(minimum, float(values.min()))
            maximum = max(maximum, float(values.max()))
            total += float(values.sum())
            total_sq += float(np.dot(values, values))
            count += int(values.size)
        region_means[start:stop] = np.asarray(block @ membership, dtype=np.float32) / region_counts
    mean = total / count
    variance = max(total_sq / count - mean * mean, 0.0)
    summary = {"min": minimum, "max": maximum, "mean": mean, "std": variance**0.5, "nan": n_nan, "inf": n_inf, "count": count}
    return summary, region_means


def lag_correlations(z: np.ndarray, labels: np.ndarray, lags: list[int]) -> pd.DataFrame:
    rows = []
    for state in STATES:
        state_mask = labels == state
        for lag in lags:
            valid = state_mask[lag:] & state_mask[:-lag]
            left = z[:-lag][valid]
            right = z[lag:][valid]
            numerator = np.sum(left * right, axis=0)
            denominator = np.sqrt(np.sum(left * left, axis=0) * np.sum(right * right, axis=0))
            corr = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
            rows.append({"state": state, "lag_steps": lag, "mean": float(np.nanmean(corr)), "median": float(np.nanmedian(corr)), "p10": float(np.nanquantile(corr, 0.1)), "p90": float(np.nanquantile(corr, 0.9)), "n_pairs": int(valid.sum())})
    return pd.DataFrame(rows)


def state_spectra(z: np.ndarray, segments: pd.DataFrame, sampling_hz: float) -> pd.DataFrame:
    rows = []
    for state in STATES:
        spectra = []
        weights = []
        for row in segments.itertuples(index=False):
            if row.state != state or row.length < 64:
                continue
            frequency, power = welch(z[row.start : row.stop], fs=sampling_hz, axis=0, nperseg=64, detrend="linear")
            spectra.append(np.nanmean(power, axis=1))
            weights.append(row.length)
        if not spectra:
            continue
        common_frequency = frequency
        average = np.average(np.stack(spectra), axis=0, weights=np.asarray(weights))
        for f, p in zip(common_frequency, average, strict=True):
            rows.append({"state": state, "frequency_hz": float(f), "mean_power": float(p)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    update_parser(parser)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data_path = args.data_dir / "data.npy"
    node_path = args.data_dir / "node_info.csv"
    time_path = args.data_dir / "time_info.csv"
    data = np.load(data_path, mmap_mode="r")
    node_info = pd.read_csv(node_path, index_col=0)
    time_info = pd.read_csv(time_path, index_col=0)
    labels = np.asarray(time_info["stimuli"].astype(str).to_numpy(), dtype="U16")
    if data.shape != (len(time_info), len(node_info)):
        raise ValueError(f"Shape mismatch: data={data.shape}, time={len(time_info)}, nodes={len(node_info)}")

    rng = np.random.default_rng(args.seed)
    segments = contiguous_segments(labels)
    segments.to_csv(args.output_dir / "segments.csv", index=False)

    region_names = np.sort(node_info["acs"].unique())
    region_lookup = {name: index for index, name in enumerate(region_names)}
    region_codes = node_info["acs"].map(region_lookup).to_numpy(dtype=np.int64)
    full_summary, region_means = exact_scan_and_region_means(data, region_codes, len(region_names), args.chunk_rows)
    np.savez_compressed(args.output_dir / "region_means.npz", values=region_means, regions=region_names, labels=labels)

    sample_index = sample_nodes_stratified(node_info, args.sample_nodes, rng)
    sample = np.asarray(data[:, sample_index], dtype=np.float32)
    sample_mean = sample.mean(axis=0, dtype=np.float64)
    sample_std = sample.std(axis=0, dtype=np.float64)
    safe_std = np.where(sample_std > 0, sample_std, 1.0)
    z = ((sample - sample_mean) / safe_std).astype(np.float32)
    np.save(args.output_dir / "sample_node_index.npy", sample_index)

    lag_table = lag_correlations(z, labels, [1, 2, 4, 8, 16, 32, 64, 128, 256])
    lag_table["seconds"] = lag_table["lag_steps"] / args.sampling_hz
    lag_table.to_csv(args.output_dir / "lag_autocorrelation.csv", index=False)
    state_spectra(z, segments, args.sampling_hz).to_csv(args.output_dir / "state_power_spectra.csv", index=False)

    pca_count = min(args.pca_nodes, z.shape[1])
    pca_columns = rng.choice(z.shape[1], size=pca_count, replace=False)
    pca = PCA(n_components=min(args.pca_components, pca_count), svd_solver="randomized", random_state=args.seed)
    scores = pca.fit_transform(z[:, pca_columns])
    np.savez_compressed(args.output_dir / "sample_pca.npz", scores=scores.astype(np.float32), explained_variance_ratio=pca.explained_variance_ratio_, sampled_columns=sample_index[pca_columns], labels=labels)

    state_rows = []
    for state in STATES:
        values = sample[labels == state]
        differences = np.diff(sample, axis=0)[(labels[:-1] == state) & (labels[1:] == state)]
        state_rows.append({"state": state, "n_time": int((labels == state).sum()), "mean": float(values.mean()), "std": float(values.std()), "q01": float(np.quantile(values, 0.01)), "median": float(np.median(values)), "q99": float(np.quantile(values, 0.99)), "dx_mean": float(differences.mean()), "dx_std": float(differences.std())})
    pd.DataFrame(state_rows).to_csv(args.output_dir / "sample_state_summary.csv", index=False)

    segment_summary = segments.groupby("state")["length"].agg(["count", "mean", "min", "median", "max"]).reset_index()
    metadata = {
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "sampling_hz": args.sampling_hz,
        "duration_seconds": (data.shape[0] - 1) / args.sampling_hz,
        "state_counts": time_info["stimuli"].value_counts().to_dict(),
        "n_segments": len(segments),
        "n_regions": len(region_names),
        "region_counts": node_info["acs"].value_counts().to_dict(),
        "full_value_summary": full_summary,
        "sample_nodes": int(len(sample_index)),
        "pca_nodes": int(pca_count),
        "pca_explained_first": float(pca.explained_variance_ratio_[0]),
        "pca_explained_first_5": float(pca.explained_variance_ratio_[:5].sum()),
        "pca_explained_first_20": float(pca.explained_variance_ratio_[:20].sum()),
        "segment_summary": segment_summary.to_dict(orient="records"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
