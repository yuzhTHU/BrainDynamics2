#!/usr/bin/env python3
"""Test whether cell-level graph context predicts detrended activity increments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


STATES = ("Awake", "NREM", "REM")


def update_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--audit-dir", type=Path, default=Path("logs/audit_data/initial"))
    parser.add_argument("--output-dir", type=Path, default=Path("logs/cell_graph_controls/initial"))
    parser.add_argument("--n-nodes", type=int, default=384)
    parser.add_argument("--histories", type=int, nargs="+", default=[4, 16])
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--highpass-seconds", type=float, default=16.0)
    parser.add_argument("--sampling-hz", type=float, default=4.146313525439001)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0, 100.0])
    parser.add_argument("--max-train-pairs", type=int, default=500000)
    parser.add_argument("--max-validation-pairs", type=int, default=250000)
    parser.add_argument("--prediction-batch-times", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261009)


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


def valid_times(table: pd.DataFrame, history: int) -> np.ndarray:
    pieces = [np.arange(row.start + history, row.stop, dtype=np.int64) for row in table.itertuples(index=False) if row.length > history]
    return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)


def table_indices(table: pd.DataFrame) -> np.ndarray:
    return np.concatenate([np.arange(row.start, row.stop, dtype=np.int64) for row in table.itertuples(index=False)])


def causal_highpass_by_episode(values: np.ndarray, segments: pd.DataFrame, width: int) -> np.ndarray:
    output = np.empty_like(values)
    for row in segments.itertuples(index=False):
        block = values[row.start : row.stop].astype(np.float64)
        cumulative = np.vstack([np.zeros((1, block.shape[1])), np.cumsum(block, axis=0)])
        stop = np.arange(1, len(block) + 1)
        start = np.maximum(0, stop - width)
        baseline = (cumulative[stop] - cumulative[start]) / (stop - start)[:, None]
        output[row.start : row.stop] = block - baseline
    return output


def row_normalized_sparse(neighbor_index: np.ndarray, neighbor_weight: np.ndarray, n_nodes: int) -> sparse.csr_matrix:
    rows = np.repeat(np.arange(n_nodes), neighbor_index.shape[1])
    columns = neighbor_index.reshape(-1)
    weights = neighbor_weight.reshape(-1)
    scale = np.sum(np.abs(neighbor_weight), axis=1)
    weights = weights / np.repeat(np.where(scale > 0, scale, 1.0), neighbor_index.shape[1])
    return sparse.csr_matrix((weights, (rows, columns)), shape=(n_nodes, n_nodes), dtype=np.float32)


def build_contexts(z: np.ndarray, train_index: np.ndarray, node_info: pd.DataFrame, neighbors: int, rng: np.random.Generator) -> tuple[dict[str, np.ndarray], dict]:
    n_nodes = z.shape[1]
    global_context = np.repeat(z.mean(axis=1, keepdims=True), n_nodes, axis=1)

    region_codes, region_names = pd.factorize(node_info["acs"], sort=True)
    region_neighbors = []
    region_weights = []
    for node in range(n_nodes):
        candidates = np.flatnonzero((region_codes == region_codes[node]) & (np.arange(n_nodes) != node))
        if not len(candidates):
            candidates = np.asarray([node])
        selected = np.resize(candidates, neighbors)
        region_neighbors.append(selected)
        region_weights.append(np.ones(neighbors))
    region_matrix = row_normalized_sparse(np.asarray(region_neighbors), np.asarray(region_weights), n_nodes)

    coordinates = node_info[["coords_2d_x", "coords_2d_y"]].to_numpy(dtype=np.float64)
    spatial_index = NearestNeighbors(n_neighbors=min(neighbors + 1, n_nodes)).fit(coordinates).kneighbors(return_distance=False)
    spatial_index = np.asarray([row[row != node][:neighbors] for node, row in enumerate(spatial_index)])
    spatial_matrix = row_normalized_sparse(spatial_index, np.ones_like(spatial_index, dtype=float), n_nodes)

    correlation = np.corrcoef(z[train_index].T)
    np.fill_diagonal(correlation, 0.0)
    correlation_index = np.argpartition(np.abs(correlation), -neighbors, axis=1)[:, -neighbors:]
    correlation_weight = np.take_along_axis(correlation, correlation_index, axis=1)
    correlation_matrix = row_normalized_sparse(correlation_index, correlation_weight, n_nodes)

    random_index = np.empty_like(correlation_index)
    for node in range(n_nodes):
        candidates = np.delete(np.arange(n_nodes), node)
        random_index[node] = rng.choice(candidates, size=neighbors, replace=False)
    sign_pool = np.sign(correlation_weight.reshape(-1))
    random_weight = rng.choice(sign_pool, size=random_index.shape, replace=True)
    random_matrix = row_normalized_sparse(random_index, random_weight, n_nodes)

    matrices = {"region": region_matrix, "spatial": spatial_matrix, "correlation": correlation_matrix, "random": random_matrix}
    contexts = {"global": global_context.astype(np.float32)}
    contexts.update({name: np.asarray((matrix @ z.T).T, dtype=np.float32) for name, matrix in matrices.items()})
    spatial_edges = set(zip(np.repeat(np.arange(n_nodes), neighbors), spatial_index.reshape(-1), strict=True))
    correlation_edges = set(zip(np.repeat(np.arange(n_nodes), neighbors), correlation_index.reshape(-1), strict=True))
    diagnostics = {
        "regions": [str(name) for name in region_names],
        "neighbors": neighbors,
        "mean_absolute_selected_correlation": float(np.mean(np.abs(correlation_weight))),
        "spatial_correlation_edge_overlap_fraction": len(spatial_edges & correlation_edges) / len(spatial_edges),
    }
    return contexts, diagnostics


def sampled_pairs(times: np.ndarray, n_nodes: int, maximum: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n_pairs = len(times) * n_nodes
    if n_pairs <= maximum:
        return np.repeat(times, n_nodes), np.tile(np.arange(n_nodes), len(times))
    flat = rng.choice(n_pairs, size=maximum, replace=False)
    return times[flat // n_nodes], flat % n_nodes


def pair_features(z: np.ndarray, context: np.ndarray | None, times: np.ndarray, nodes: np.ndarray, history: int) -> np.ndarray:
    self_features = np.column_stack([z[times - lag, nodes] for lag in range(1, history + 1)])
    if context is None:
        return self_features
    context_features = np.column_stack([context[times - lag, nodes] for lag in range(1, history + 1)])
    return np.concatenate([self_features, context_features], axis=1)


def fit_model(z: np.ndarray, context: np.ndarray | None, split: dict[str, pd.DataFrame], history: int, alphas: list[float], train_max: int, validation_max: int, rng: np.random.Generator) -> tuple[StandardScaler, Ridge, float, float]:
    train_times, train_nodes = sampled_pairs(valid_times(split["train"], history), z.shape[1], train_max, rng)
    validation_times, validation_nodes = sampled_pairs(valid_times(split["validation"], history), z.shape[1], validation_max, rng)
    x_train = pair_features(z, context, train_times, train_nodes, history)
    x_validation = pair_features(z, context, validation_times, validation_nodes, history)
    y_train = z[train_times, train_nodes] - z[train_times - 1, train_nodes]
    y_validation = z[validation_times, validation_nodes] - z[validation_times - 1, validation_nodes]
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_validation = scaler.transform(x_validation)
    best_alpha = None
    best_loss = np.inf
    for alpha in alphas:
        estimator = Ridge(alpha=alpha).fit(x_train, y_train)
        loss = float(np.mean((estimator.predict(x_validation) - y_validation) ** 2))
        if loss < best_loss:
            best_alpha, best_loss = alpha, loss
    estimator = Ridge(alpha=best_alpha).fit(x_train, y_train)
    return scaler, estimator, float(best_alpha), best_loss


def evaluate_model(z: np.ndarray, context: np.ndarray | None, table: pd.DataFrame, history: int, scaler: StandardScaler | None, estimator: Ridge | None, batch_times: int) -> dict[str, float]:
    times = valid_times(table, history)
    truth_parts = []
    prediction_parts = []
    for start in range(0, len(times), batch_times):
        batch = times[start : start + batch_times]
        pair_times = np.repeat(batch, z.shape[1])
        pair_nodes = np.tile(np.arange(z.shape[1]), len(batch))
        truth = z[pair_times, pair_nodes] - z[pair_times - 1, pair_nodes]
        if estimator is None:
            prediction = np.zeros_like(truth)
        else:
            features = pair_features(z, context, pair_times, pair_nodes, history)
            prediction = estimator.predict(scaler.transform(features))
        truth_parts.append(truth.reshape(len(batch), z.shape[1]))
        prediction_parts.append(prediction.reshape(len(batch), z.shape[1]))
    truth = np.concatenate(truth_parts)
    prediction = np.concatenate(prediction_parts)
    node_r2 = [r2_score(truth[:, node], prediction[:, node]) for node in range(z.shape[1])]
    node_corr = [np.corrcoef(truth[:, node], prediction[:, node])[0, 1] if np.std(prediction[:, node]) > 0 and np.std(truth[:, node]) > 0 else np.nan for node in range(z.shape[1])]
    finite_node_corr = np.asarray(node_corr)[np.isfinite(node_corr)]
    return {"dx_r2_flat": float(r2_score(truth.reshape(-1), prediction.reshape(-1))), "dx_corr_flat": float(np.corrcoef(truth.reshape(-1), prediction.reshape(-1))[0, 1]) if np.std(prediction) > 0 else np.nan, "dx_r2_node_median": float(np.nanmedian(node_r2)), "dx_corr_node_median": float(np.median(finite_node_corr)) if len(finite_node_corr) else np.nan, "n_test_times": len(times)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    update_parser(parser)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    audit_index = np.load(args.audit_dir / "sample_node_index.npy")
    selected = np.sort(rng.choice(audit_index, size=min(args.n_nodes, len(audit_index)), replace=False))
    data = np.load(args.data_dir / "data.npy", mmap_mode="r")
    values = np.asarray(data[:, selected], dtype=np.float32)
    all_node_info = pd.read_csv(args.data_dir / "node_info.csv", index_col=0)
    node_info = all_node_info.iloc[selected].reset_index(drop=True)
    labels = pd.read_csv(args.data_dir / "time_info.csv", index_col=0)["stimuli"].astype(str).to_numpy()
    segments = make_segments(labels)
    width = max(3, int(round(args.highpass_seconds * args.sampling_hz)))
    width += 1 - width % 2
    residual = causal_highpass_by_episode(values, segments, width)
    rows = []
    diagnostics = {}
    for state in STATES:
        split = split_segments(segments, state)
        train_index = table_indices(split["train"])
        mean = residual[train_index].mean(axis=0, dtype=np.float64)
        std = residual[train_index].std(axis=0, dtype=np.float64)
        z = ((residual - mean) / np.where(std > 0, std, 1.0)).astype(np.float32)
        contexts, state_diagnostics = build_contexts(z, train_index, node_info, args.neighbors, rng)
        diagnostics[state] = state_diagnostics
        for history in args.histories:
            rows.append({"state": state, "history": history, "model": "persistence", "alpha": 0.0, "validation_mse": np.nan, **evaluate_model(z, None, split["test"], history, None, None, args.prediction_batch_times)})
            for model, context_name in (("self", None), ("self_global", "global"), ("self_region", "region"), ("self_spatial", "spatial"), ("self_correlation", "correlation"), ("self_random", "random")):
                context = contexts.get(context_name) if context_name else None
                scaler, estimator, alpha, validation_loss = fit_model(z, context, split, history, args.alphas, args.max_train_pairs, args.max_validation_pairs, rng)
                rows.append({"state": state, "history": history, "model": model, "alpha": alpha, "validation_mse": validation_loss, **evaluate_model(z, context, split["test"], history, scaler, estimator, args.prediction_batch_times)})
    pd.DataFrame(rows).to_csv(args.output_dir / "model_summary.csv", index=False)
    np.save(args.output_dir / "sample_node_index.npy", selected)
    (args.output_dir / "graph_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
