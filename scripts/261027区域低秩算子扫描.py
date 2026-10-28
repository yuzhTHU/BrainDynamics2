#!/usr/bin/env python3
"""Select the rank of regional linear increment operators on held-out episodes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


STATES = ("Awake", "NREM", "REM")


def update_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--audit-dir", type=Path, default=Path("logs/audit_data/initial"))
    parser.add_argument("--output-dir", type=Path, default=Path("logs/regional_lowrank_sweep/initial"))
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 3, 5, 8, 12, 25])
    parser.add_argument("--structures", nargs="+", choices=("pure_lowrank", "diagonal_plus_lowrank"), default=["pure_lowrank", "diagonal_plus_lowrank"])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
    parser.add_argument("--highpass-seconds", type=float, default=16.0)
    parser.add_argument("--sampling-hz", type=float, default=4.146313525439001)
    parser.add_argument("--max-rollout-steps", type=int, default=80)
    parser.add_argument("--rollout-stride", type=int, default=40)


def make_segments(labels: np.ndarray) -> pd.DataFrame:
    changes = np.r_[True, labels[1:] != labels[:-1], True]
    boundaries = np.flatnonzero(changes)
    return pd.DataFrame([{"start": int(a), "stop": int(b), "length": int(b - a), "state": str(labels[a])} for a, b in zip(boundaries[:-1], boundaries[1:], strict=True)])


def split_segments(segments: pd.DataFrame, state: str) -> dict[str, pd.DataFrame]:
    selected = segments[segments.state == state].reset_index(drop=True)
    n_train = max(1, int(np.floor(0.6 * len(selected))))
    n_validation = max(1, int(np.floor(0.2 * len(selected))))
    if n_train + n_validation >= len(selected):
        n_train = len(selected) - 2
        n_validation = 1
    return {"train": selected.iloc[:n_train], "validation": selected.iloc[n_train : n_train + n_validation], "test": selected.iloc[n_train + n_validation :]}


def table_indices(table: pd.DataFrame) -> np.ndarray:
    return np.concatenate([np.arange(row.start, row.stop, dtype=np.int64) for row in table.itertuples(index=False)])


def targets(table: pd.DataFrame) -> np.ndarray:
    return np.concatenate([np.arange(row.start + 1, row.stop, dtype=np.int64) for row in table.itertuples(index=False) if row.length > 1])


def causal_highpass(values: np.ndarray, segments: pd.DataFrame, width: int) -> np.ndarray:
    output = np.empty_like(values, dtype=np.float64)
    for row in segments.itertuples(index=False):
        block = values[row.start : row.stop]
        cumulative = np.vstack([np.zeros((1, block.shape[1])), np.cumsum(block, axis=0)])
        stop = np.arange(1, len(block) + 1)
        start = np.maximum(0, stop - width)
        output[row.start : row.stop] = block - (cumulative[stop] - cumulative[start]) / (stop - start)[:, None]
    return output


def truncate(matrix: np.ndarray, rank: int, structure: str) -> np.ndarray:
    diagonal = np.diag(np.diag(matrix)) if structure == "diagonal_plus_lowrank" else np.zeros_like(matrix)
    residual = matrix - diagonal
    u, singular, vt = np.linalg.svd(residual, full_matrices=False)
    rank = min(rank, len(singular))
    return diagonal + (u[:, :rank] * singular[:rank]) @ vt[:rank]


def predict_dx(x: np.ndarray, operator: np.ndarray, intercept: np.ndarray) -> np.ndarray:
    return x @ operator.T + intercept


def rollout(operator: np.ndarray, intercept: np.ndarray, z: np.ndarray, table: pd.DataFrame, max_steps: int, stride: int, sampling_hz: float) -> pd.DataFrame:
    buckets: dict[int, dict[str, list[np.ndarray]]] = {}
    for segment in table.itertuples(index=False):
        for origin in range(segment.start, segment.stop - 1, stride):
            simulated = z[origin].copy()
            for step in range(1, min(max_steps, segment.stop - origin - 1) + 1):
                simulated = simulated + predict_dx(simulated[None], operator, intercept)[0]
                bucket = buckets.setdefault(step, {"truth": [], "prediction": []})
                bucket["truth"].append(z[origin + step])
                bucket["prediction"].append(simulated.copy())
    rows = []
    for step, bucket in sorted(buckets.items()):
        truth, prediction = np.stack(bucket["truth"]), np.stack(bucket["prediction"])
        rows.append({"step": step, "seconds": step / sampling_hz, "r2_flat": float(r2_score(truth.ravel(), prediction.ravel())), "pearson_flat": float(np.corrcoef(truth.ravel(), prediction.ravel())[0, 1]), "n_origins": len(truth)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    update_parser(parser)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = np.load(args.audit_dir / "region_means.npz", allow_pickle=True)
    values = archive["values"].astype(np.float64)
    labels = archive["labels"].astype(str)
    segments = make_segments(labels)
    width = max(3, int(round(args.highpass_seconds * args.sampling_hz)))
    values = causal_highpass(values, segments, width)
    summaries, rollout_tables, singular_rows = [], [], []
    for state in STATES:
        split = split_segments(segments, state)
        train_index = table_indices(split["train"])
        mean, std = values[train_index].mean(axis=0), values[train_index].std(axis=0)
        z = (values - mean) / np.where(std > 0, std, 1.0)
        train, validation, test = targets(split["train"]), targets(split["validation"]), targets(split["test"])
        x_train, y_train = z[train - 1], z[train] - z[train - 1]
        x_validation, y_validation = z[validation - 1], z[validation] - z[validation - 1]
        x_test, y_test = z[test - 1], z[test] - z[test - 1]
        candidates = {}
        for alpha in args.alphas:
            estimator = Ridge(alpha=alpha).fit(x_train, y_train)
            candidates[alpha] = (estimator.coef_, estimator.intercept_)
        full = candidates[args.alphas[0]][0]
        singular = np.linalg.svd(full, compute_uv=False)
        for index, value in enumerate(singular, start=1):
            singular_rows.append({"state": state, "component": index, "singular_value": value, "cumulative_energy": float(np.sum(singular[:index] ** 2) / np.sum(singular**2))})
        for structure in args.structures:
            for rank in args.ranks:
                best = None
                for alpha, (operator, intercept) in candidates.items():
                    reduced = truncate(operator, rank, structure)
                    prediction = predict_dx(x_validation, reduced, intercept)
                    loss = float(np.mean((prediction - y_validation) ** 2))
                    if best is None or loss < best[0]:
                        best = (loss, alpha, reduced, intercept)
                validation_loss, alpha, operator, intercept = best
                prediction = predict_dx(x_test, operator, intercept)
                node_r2 = [r2_score(y_test[:, node], prediction[:, node]) for node in range(z.shape[1])]
                curve = rollout(operator, intercept, z, split["test"], args.max_rollout_steps, args.rollout_stride, args.sampling_hz)
                curve.insert(0, "rank", rank)
                curve.insert(0, "structure", structure)
                curve.insert(0, "state", state)
                rollout_tables.append(curve)
                at_four = curve.iloc[(curve.seconds - 4.0).abs().argmin()]
                summaries.append({"state": state, "structure": structure, "rank": rank, "alpha": alpha, "validation_mse": validation_loss, "dx_r2_flat": float(r2_score(y_test.ravel(), prediction.ravel())), "dx_r2_region_median": float(np.median(node_r2)), "dx_corr_flat": float(np.corrcoef(y_test.ravel(), prediction.ravel())[0, 1]), "rollout_r2_at_4s": at_four.r2_flat, "rollout_corr_at_4s": at_four.pearson_flat})
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output_dir / "rank_summary.csv", index=False)
    pd.concat(rollout_tables, ignore_index=True).to_csv(args.output_dir / "rollout_metrics.csv", index=False)
    pd.DataFrame(singular_rows).to_csv(args.output_dir / "singular_spectra.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
