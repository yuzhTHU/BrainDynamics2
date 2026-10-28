#!/usr/bin/env python3
"""Validate low-rank regional operators by joint fitting, transfer, and reversal."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import detrend
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


STATES = ("Awake", "NREM", "REM")


def update_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--audit-dir", type=Path, default=Path("logs/audit_data/initial"))
    parser.add_argument("--output-dir", type=Path, default=Path("logs/regional_operator_validation/initial"))
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 3, 5, 8, 12, 25])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.1, 1.0, 10.0, 100.0, 1000.0])
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--highpass-seconds", type=float, default=16.0)
    parser.add_argument("--sampling-hz", type=float, default=4.146313525439001)


def make_segments(labels: np.ndarray) -> pd.DataFrame:
    changes = np.r_[True, labels[1:] != labels[:-1], True]
    boundaries = np.flatnonzero(changes)
    return pd.DataFrame([{"start": int(a), "stop": int(b), "state": str(labels[a])} for a, b in zip(boundaries[:-1], boundaries[1:], strict=True)])


def splits(segments: pd.DataFrame) -> dict[str, dict[str, pd.DataFrame]]:
    result = {}
    for state in STATES:
        table = segments[segments.state == state].reset_index(drop=True)
        n_train = max(1, int(np.floor(0.6 * len(table))))
        n_validation = max(1, int(np.floor(0.2 * len(table))))
        if n_train + n_validation >= len(table):
            n_train, n_validation = len(table) - 2, 1
        result[state] = {"train": table.iloc[:n_train], "validation": table.iloc[n_train : n_train + n_validation], "test": table.iloc[n_train + n_validation :]}
    return result


def indices(table: pd.DataFrame, targets: bool = False) -> np.ndarray:
    offset = 1 if targets else 0
    return np.concatenate([np.arange(row.start + offset, row.stop, dtype=np.int64) for row in table.itertuples(index=False) if row.stop - row.start > offset])


def causal_highpass(values: np.ndarray, segments: pd.DataFrame, width: int) -> np.ndarray:
    output = np.empty_like(values)
    for row in segments.itertuples(index=False):
        block = values[row.start : row.stop]
        cumulative = np.vstack([np.zeros((1, block.shape[1])), np.cumsum(block, axis=0)])
        stop = np.arange(1, len(block) + 1)
        start = np.maximum(0, stop - width)
        output[row.start : row.stop] = block - (cumulative[stop] - cumulative[start]) / (stop - start)[:, None]
    return output


def linear_detrend(values: np.ndarray, segments: pd.DataFrame) -> np.ndarray:
    output = np.empty_like(values)
    for row in segments.itertuples(index=False):
        output[row.start : row.stop] = detrend(values[row.start : row.stop], axis=0, type="linear")
    return output


def reverse_episodes(values: np.ndarray, segments: pd.DataFrame) -> np.ndarray:
    output = values.copy()
    for row in segments.itertuples(index=False):
        output[row.start : row.stop] = values[row.start : row.stop][::-1]
    return output


def truncate(matrix: np.ndarray, rank: int) -> np.ndarray:
    if rank >= min(matrix.shape):
        return matrix
    u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    return (u[:, :rank] * singular[:rank]) @ vt[:rank]


def fit_operator(x: np.ndarray, y: np.ndarray, alpha: float, rank: int, iterations: int) -> tuple[np.ndarray, np.ndarray]:
    x_mean, y_mean = x.mean(axis=0), y.mean(axis=0)
    xc, yc = x - x_mean, y - y_mean
    diagonal = np.zeros(x.shape[1])
    lowrank = np.zeros((y.shape[1], x.shape[1]))
    for _ in range(iterations):
        residual = yc - xc * diagonal
        dense = Ridge(alpha=alpha, fit_intercept=False).fit(xc, residual).coef_
        lowrank = truncate(dense, rank)
        residual = yc - xc @ lowrank.T
        diagonal = np.sum(xc * residual, axis=0) / (np.sum(xc * xc, axis=0) + alpha)
    operator = lowrank + np.diag(diagonal)
    intercept = y_mean - x_mean @ operator.T
    return operator, intercept


def select_operator(z: np.ndarray, train: np.ndarray, validation: np.ndarray, rank: int, alphas: list[float], iterations: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    x_train, y_train = z[train - 1], z[train] - z[train - 1]
    x_validation, y_validation = z[validation - 1], z[validation] - z[validation - 1]
    best = None
    for alpha in alphas:
        operator, intercept = fit_operator(x_train, y_train, alpha, rank, iterations)
        prediction = x_validation @ operator.T + intercept
        loss = float(np.mean((prediction - y_validation) ** 2))
        if best is None or loss < best[0]:
            best = loss, operator, intercept, alpha
    return best[1], best[2], float(best[3]), best[0]


def evaluate(z: np.ndarray, test: np.ndarray, operator: np.ndarray, intercept: np.ndarray) -> dict[str, float]:
    truth = z[test] - z[test - 1]
    prediction = z[test - 1] @ operator.T + intercept
    node_r2 = [r2_score(truth[:, node], prediction[:, node]) for node in range(z.shape[1])]
    return {"dx_r2_flat": float(r2_score(truth.ravel(), prediction.ravel())), "dx_r2_region_median": float(np.median(node_r2)), "dx_corr_flat": float(np.corrcoef(truth.ravel(), prediction.ravel())[0, 1]), "n_targets": len(test)}


def normalize(values: np.ndarray, train_index: np.ndarray) -> np.ndarray:
    mean, std = values[train_index].mean(axis=0), values[train_index].std(axis=0)
    return (values - mean) / np.where(std > 0, std, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    update_parser(parser)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = np.load(args.audit_dir / "region_means.npz", allow_pickle=True)
    raw, labels = archive["values"].astype(np.float64), archive["labels"].astype(str)
    segments = make_segments(labels)
    split = splits(segments)
    width = max(3, int(round(args.highpass_seconds * args.sampling_hz)))
    highpass = causal_highpass(raw, segments, width)

    joint_rows = []
    for state in STATES:
        z = normalize(highpass, indices(split[state]["train"]))
        train, validation, test = (indices(split[state][part], targets=True) for part in ("train", "validation", "test"))
        for rank in args.ranks:
            operator, intercept, alpha, validation_loss = select_operator(z, train, validation, rank, args.alphas, args.iterations)
            joint_rows.append({"state": state, "rank": rank, "alpha": alpha, "validation_mse": validation_loss, **evaluate(z, test, operator, intercept)})
    pd.DataFrame(joint_rows).to_csv(args.output_dir / "joint_rank_summary.csv", index=False)

    shared_train_index = np.concatenate([indices(split[state]["train"]) for state in STATES])
    z_shared = normalize(highpass, shared_train_index)
    transfer_rows = []
    sources = list(STATES) + ["Pooled"]
    for source in sources:
        source_states = STATES if source == "Pooled" else (source,)
        train = np.concatenate([indices(split[state]["train"], targets=True) for state in source_states])
        validation = np.concatenate([indices(split[state]["validation"], targets=True) for state in source_states])
        for rank in (5, 25):
            operator, intercept, alpha, validation_loss = select_operator(z_shared, train, validation, rank, args.alphas, args.iterations)
            for target_state in STATES:
                test = indices(split[target_state]["test"], targets=True)
                transfer_rows.append({"source_state": source, "target_state": target_state, "rank": rank, "alpha": alpha, "validation_mse": validation_loss, **evaluate(z_shared, test, operator, intercept)})
    pd.DataFrame(transfer_rows).to_csv(args.output_dir / "cross_state_transfer.csv", index=False)

    detrended = linear_detrend(raw, segments)
    reversal_rows = []
    for direction, values in (("forward", detrended), ("reversed", reverse_episodes(detrended, segments))):
        for state in STATES:
            z = normalize(values, indices(split[state]["train"]))
            train, validation, test = (indices(split[state][part], targets=True) for part in ("train", "validation", "test"))
            for rank in (5, 25):
                operator, intercept, alpha, validation_loss = select_operator(z, train, validation, rank, args.alphas, args.iterations)
                reversal_rows.append({"direction": direction, "state": state, "rank": rank, "alpha": alpha, "validation_mse": validation_loss, **evaluate(z, test, operator, intercept)})
    pd.DataFrame(reversal_rows).to_csv(args.output_dir / "time_reversal.csv", index=False)
    print(pd.DataFrame(joint_rows).to_string(index=False))
    print(pd.DataFrame(transfer_rows).to_string(index=False))
    print(pd.DataFrame(reversal_rows).to_string(index=False))


if __name__ == "__main__":
    main()
