#!/usr/bin/env python3
"""Compare local-feedback, diffusion, and advection terms on spatial-bin dynamics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


STATES = ("Awake", "NREM")
MODELS = ("zero", "local", "local_diffusion", "local_advection", "local_advection_diffusion")


def update_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--event-arrays", type=Path, default=Path("logs/spatial_bin_event_propagation/initial/event_arrays.npz"))
    parser.add_argument("--time-info", type=Path, default=Path("data/processed/time_info.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("logs/spatial_pde_library/initial"))
    parser.add_argument("--neighbors", type=int, default=6)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
    parser.add_argument("--reverse-time", action="store_true")


def make_segments(labels: np.ndarray) -> pd.DataFrame:
    changes = np.r_[True, labels[1:] != labels[:-1], True]
    boundaries = np.flatnonzero(changes)
    return pd.DataFrame([{"start": int(a), "stop": int(b), "state": str(labels[a])} for a, b in zip(boundaries[:-1], boundaries[1:], strict=True)])


def split_segments(segments: pd.DataFrame, state: str) -> dict[str, pd.DataFrame]:
    selected = segments[segments.state == state].reset_index(drop=True)
    n_train = max(1, int(np.floor(0.6 * len(selected))))
    n_validation = max(1, int(np.floor(0.2 * len(selected))))
    return {"train": selected.iloc[:n_train], "validation": selected.iloc[n_train : n_train + n_validation], "test": selected.iloc[n_train + n_validation :]}


def indices(table: pd.DataFrame, targets: bool = False) -> np.ndarray:
    offset = 1 if targets else 0
    return np.concatenate([np.arange(row.start + offset, row.stop, dtype=np.int64) for row in table.itertuples(index=False) if row.stop - row.start > offset])


def reverse_episodes(values: np.ndarray, segments: pd.DataFrame) -> np.ndarray:
    output = values.copy()
    for row in segments.itertuples(index=False):
        output[row.start : row.stop] = values[row.start : row.stop][::-1]
    return output


def spatial_operators(centers: np.ndarray, neighbors: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normalized = (centers - centers.mean(axis=0)) / centers.std(axis=0)
    neighbor_index = NearestNeighbors(n_neighbors=neighbors + 1).fit(normalized).kneighbors(return_distance=False)
    n_bins = len(centers)
    laplacian, gradient_x, gradient_y = np.zeros((n_bins, n_bins)), np.zeros((n_bins, n_bins)), np.zeros((n_bins, n_bins))
    for target, row in enumerate(neighbor_index):
        selected = row[row != target][:neighbors]
        offsets = normalized[selected] - normalized[target]
        inverse = np.linalg.pinv(offsets)
        gradient_x[target, selected], gradient_y[target, selected] = inverse[0], inverse[1]
        gradient_x[target, target], gradient_y[target, target] = -inverse[0].sum(), -inverse[1].sum()
        weights = 1.0 / np.maximum(np.sum(offsets**2, axis=1), 1e-6)
        weights /= weights.sum()
        laplacian[target, selected] = weights
        laplacian[target, target] = -1.0
    return laplacian, gradient_x, gradient_y


def design(z: np.ndarray, targets: np.ndarray, operators: tuple[np.ndarray, np.ndarray, np.ndarray], model: str) -> tuple[np.ndarray, list[str]]:
    current = z[targets - 1]
    n_times, n_bins = current.shape
    bins = np.tile(np.arange(n_bins), n_times)
    flat_current = current.reshape(-1)
    identity = np.zeros((n_times * n_bins, n_bins), dtype=np.float32)
    identity[np.arange(len(identity)), bins] = 1.0
    local = identity * flat_current[:, None]
    blocks, names = [identity, local], [*[f"intercept_bin_{i}" for i in range(n_bins)], *[f"self_bin_{i}" for i in range(n_bins)]]
    laplacian, gradient_x, gradient_y = operators
    if model in {"local_diffusion", "local_advection_diffusion"}:
        blocks.append((current @ laplacian.T).reshape(-1, 1))
        names.append("diffusion")
    if model in {"local_advection", "local_advection_diffusion"}:
        blocks.extend([(current @ gradient_x.T).reshape(-1, 1), (current @ gradient_y.T).reshape(-1, 1)])
        names.extend(["gradient_x", "gradient_y"])
    return np.concatenate(blocks, axis=1), names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    update_parser(parser)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = np.load(args.event_arrays)
    values, centers = archive["binned_detrended"].astype(np.float64), archive["centers"].astype(np.float64)
    labels = pd.read_csv(args.time_info, index_col=0)["stimuli"].astype(str).to_numpy()
    segments = make_segments(labels)
    if args.reverse_time:
        values = reverse_episodes(values, segments)
    operators = spatial_operators(centers, args.neighbors)
    rows, coefficients = [], []
    for state in STATES:
        split = split_segments(segments, state)
        train_frames = indices(split["train"])
        mean, std = values[train_frames].mean(axis=0), values[train_frames].std(axis=0)
        z = (values - mean) / np.where(std > 0, std, 1.0)
        train, validation, test = (indices(split[part], targets=True) for part in ("train", "validation", "test"))
        y_train = (z[train] - z[train - 1]).reshape(-1)
        y_validation = (z[validation] - z[validation - 1]).reshape(-1)
        y_test_matrix = z[test] - z[test - 1]
        for model in MODELS:
            if model == "zero":
                prediction = np.zeros_like(y_test_matrix)
                rows.append({"state": state, "model": model, "alpha": 0.0, "validation_mse": float(np.mean(y_validation**2)), "dx_r2_flat": float(r2_score(y_test_matrix.ravel(), prediction.ravel())), "dx_r2_bin_median": float(np.median([r2_score(y_test_matrix[:, index], prediction[:, index]) for index in range(z.shape[1])]))})
                continue
            x_train, names = design(z, train, operators, model)
            x_validation, _ = design(z, validation, operators, model)
            x_test, _ = design(z, test, operators, model)
            scaler = StandardScaler().fit(x_train)
            best = None
            for alpha in args.alphas:
                estimator = Ridge(alpha=alpha).fit(scaler.transform(x_train), y_train)
                loss = float(np.mean((estimator.predict(scaler.transform(x_validation)) - y_validation) ** 2))
                if best is None or loss < best[0]:
                    best = loss, alpha, estimator
            prediction = best[2].predict(scaler.transform(x_test)).reshape(y_test_matrix.shape)
            bin_r2 = [r2_score(y_test_matrix[:, index], prediction[:, index]) for index in range(z.shape[1])]
            rows.append({"state": state, "model": model, "alpha": best[1], "validation_mse": best[0], "dx_r2_flat": float(r2_score(y_test_matrix.ravel(), prediction.ravel())), "dx_r2_bin_median": float(np.median(bin_r2))})
            raw_coefficient = best[2].coef_ / scaler.scale_
            for name, coefficient in zip(names, raw_coefficient, strict=True):
                coefficients.append({"state": state, "model": model, "term": name, "coefficient_per_frame": coefficient})
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "model_summary.csv", index=False)
    pd.DataFrame(coefficients).to_csv(args.output_dir / "coefficients.csv", index=False)
    np.savez_compressed(args.output_dir / "spatial_operators.npz", laplacian=operators[0], gradient_x=operators[1], gradient_y=operators[2], centers=centers)
    print(summary.to_string(index=False))
    print(pd.DataFrame(coefficients)[pd.DataFrame(coefficients).term.isin(["diffusion", "gradient_x", "gradient_y"])].to_string(index=False))


if __name__ == "__main__":
    main()
