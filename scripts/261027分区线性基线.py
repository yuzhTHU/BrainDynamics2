#!/usr/bin/env python3
"""Compare regional self-history, global-context, and full VAR baselines."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


STATES = ("Awake", "NREM", "REM")


def update_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--audit-dir", type=Path, default=Path("logs/audit_data/initial"))
    parser.add_argument("--output-dir", type=Path, default=Path("logs/regional_linear_baselines/initial"))
    parser.add_argument("--histories", type=int, nargs="+", default=[1, 3, 10, 40])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
    parser.add_argument("--max-rollout-steps", type=int, default=80)
    parser.add_argument("--rollout-stride", type=int, default=40)
    parser.add_argument("--sampling-hz", type=float, default=4.146313525439001)


def make_segments(labels: np.ndarray) -> pd.DataFrame:
    changes = np.r_[True, labels[1:] != labels[:-1], True]
    boundaries = np.flatnonzero(changes)
    return pd.DataFrame([{"start": int(a), "stop": int(b), "length": int(b - a), "state": str(labels[a])} for a, b in zip(boundaries[:-1], boundaries[1:], strict=True)])


def split_segments(segments: pd.DataFrame, state: str) -> dict[str, pd.DataFrame]:
    selected = segments[segments.state == state].reset_index(drop=True)
    n = len(selected)
    if n < 3:
        raise ValueError(f"Need at least three {state} segments, found {n}")
    n_train = max(1, int(np.floor(0.6 * n)))
    n_val = max(1, int(np.floor(0.2 * n)))
    if n_train + n_val >= n:
        n_train = n - 2
        n_val = 1
    return {"train": selected.iloc[:n_train], "val": selected.iloc[n_train : n_train + n_val], "test": selected.iloc[n_train + n_val :]}


def valid_targets(segment_table: pd.DataFrame, history: int) -> np.ndarray:
    pieces = [np.arange(row.start + history, row.stop, dtype=np.int64) for row in segment_table.itertuples(index=False) if row.length > history]
    return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)


def history_tensor(z: np.ndarray, targets: np.ndarray, history: int) -> np.ndarray:
    return np.stack([z[targets - lag] for lag in range(1, history + 1)], axis=1)


@dataclass
class FittedModel:
    kind: str
    history: int
    estimators: list[Ridge] | Ridge | None

    def predict(self, history_values: np.ndarray) -> np.ndarray:
        # history_values has shape (batch, history, region), newest first.
        if self.kind == "persistence":
            return history_values[:, 0]
        if self.kind == "self":
            output = np.empty((len(history_values), history_values.shape[2]))
            for region, estimator in enumerate(self.estimators):
                output[:, region] = estimator.predict(history_values[:, :, region])
            return output
        if self.kind == "self_global":
            global_history = history_values.mean(axis=2)
            output = np.empty((len(history_values), history_values.shape[2]))
            for region, estimator in enumerate(self.estimators):
                features = np.concatenate([history_values[:, :, region], global_history], axis=1)
                output[:, region] = estimator.predict(features)
            return output
        if self.kind == "full_var":
            return self.estimators.predict(history_values.reshape(len(history_values), -1))
        raise ValueError(self.kind)


def build_features(history_values: np.ndarray, kind: str, region: int | None = None) -> np.ndarray:
    if kind == "self":
        return history_values[:, :, region]
    if kind == "self_global":
        return np.concatenate([history_values[:, :, region], history_values.mean(axis=2)], axis=1)
    if kind == "full_var":
        return history_values.reshape(len(history_values), -1)
    raise ValueError(kind)


def select_and_fit(z: np.ndarray, split: dict[str, pd.DataFrame], history: int, kind: str, alphas: list[float]) -> tuple[FittedModel, float, float]:
    train_targets = valid_targets(split["train"], history)
    val_targets = valid_targets(split["val"], history)
    train_history = history_tensor(z, train_targets, history)
    val_history = history_tensor(z, val_targets, history)
    y_train = z[train_targets]
    y_val = z[val_targets]
    if kind == "persistence":
        prediction = val_history[:, 0]
        return FittedModel(kind, history, None), 0.0, float(np.mean((prediction - y_val) ** 2))

    best_alpha = None
    best_loss = np.inf
    if kind in {"self", "self_global"}:
        for alpha in alphas:
            losses = []
            for region in range(z.shape[1]):
                estimator = Ridge(alpha=alpha, fit_intercept=True)
                estimator.fit(build_features(train_history, kind, region), y_train[:, region])
                prediction = estimator.predict(build_features(val_history, kind, region))
                losses.append(np.mean((prediction - y_val[:, region]) ** 2))
            loss = float(np.mean(losses))
            if loss < best_loss:
                best_alpha, best_loss = alpha, loss
        estimators = []
        for region in range(z.shape[1]):
            estimator = Ridge(alpha=best_alpha, fit_intercept=True)
            estimator.fit(build_features(train_history, kind, region), y_train[:, region])
            estimators.append(estimator)
        return FittedModel(kind, history, estimators), float(best_alpha), best_loss

    for alpha in alphas:
        estimator = Ridge(alpha=alpha, fit_intercept=True)
        estimator.fit(build_features(train_history, kind), y_train)
        prediction = estimator.predict(build_features(val_history, kind))
        loss = float(np.mean((prediction - y_val) ** 2))
        if loss < best_loss:
            best_alpha, best_loss = alpha, loss
    estimator = Ridge(alpha=best_alpha, fit_intercept=True)
    estimator.fit(build_features(train_history, kind), y_train)
    return FittedModel(kind, history, estimator), float(best_alpha), best_loss


def flattened_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a = y_true.reshape(-1)
    b = y_pred.reshape(-1)
    return float(np.corrcoef(a, b)[0, 1])


def one_step_metrics(model: FittedModel, z: np.ndarray, segment_table: pd.DataFrame) -> dict[str, float]:
    targets = valid_targets(segment_table, model.history)
    history_values = history_tensor(z, targets, model.history)
    truth = z[targets]
    prediction = model.predict(history_values)
    node_r2 = np.asarray([r2_score(truth[:, k], prediction[:, k]) for k in range(z.shape[1])])
    node_corr = np.asarray([np.corrcoef(truth[:, k], prediction[:, k])[0, 1] for k in range(z.shape[1])])
    dx_truth = truth - history_values[:, 0]
    dx_prediction = prediction - history_values[:, 0]
    return {
        "one_step_r2_flat": float(r2_score(truth.reshape(-1), prediction.reshape(-1))),
        "one_step_corr_flat": flattened_pearson(truth, prediction),
        "one_step_r2_region_median": float(np.nanmedian(node_r2)),
        "one_step_corr_region_median": float(np.nanmedian(node_corr)),
        "dx_r2_flat": float(r2_score(dx_truth.reshape(-1), dx_prediction.reshape(-1))),
        "n_test_targets": int(len(targets)),
    }


def rollout_metrics(model: FittedModel, z: np.ndarray, segment_table: pd.DataFrame, max_steps: int, stride: int, sampling_hz: float) -> pd.DataFrame:
    records: dict[int, dict[str, list[np.ndarray]]] = {}
    for segment in segment_table.itertuples(index=False):
        latest_origin = segment.stop - 2
        for origin in range(segment.start + model.history - 1, latest_origin + 1, stride):
            available = min(max_steps, segment.stop - origin - 1)
            simulated = [z[origin - lag].copy() for lag in range(model.history)]
            simulated = simulated[::-1]
            # simulated is oldest to newest for convenient append.
            for step in range(1, available + 1):
                recent = np.stack(simulated[-model.history:][::-1], axis=0)[None]
                prediction = model.predict(recent)[0]
                simulated.append(prediction)
                bucket = records.setdefault(step, {"truth": [], "prediction": []})
                bucket["truth"].append(z[origin + step])
                bucket["prediction"].append(prediction)
    rows = []
    for step, bucket in sorted(records.items()):
        truth = np.stack(bucket["truth"])
        prediction = np.stack(bucket["prediction"])
        region_r2 = np.asarray([r2_score(truth[:, k], prediction[:, k]) for k in range(z.shape[1])]) if len(truth) > 1 else np.full(z.shape[1], np.nan)
        rows.append({"step": step, "seconds": step / sampling_hz, "pearson_flat": flattened_pearson(truth, prediction), "r2_flat": float(r2_score(truth.reshape(-1), prediction.reshape(-1))), "r2_region_median": float(np.nanmedian(region_r2)), "n_origins": len(truth)})
    return pd.DataFrame(rows)


def threshold_horizon(table: pd.DataFrame, column: str, threshold: float) -> float:
    ordered = table.sort_values("step")
    passing = ordered[column].to_numpy() > threshold
    first_failure = np.flatnonzero(~passing)
    stop = int(first_failure[0]) if len(first_failure) else len(ordered)
    return float(ordered.iloc[stop - 1].seconds) if stop else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    update_parser(parser)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    archive = np.load(args.audit_dir / "region_means.npz", allow_pickle=True)
    values = archive["values"].astype(np.float64)
    labels = archive["labels"].astype(str)
    region_names = archive["regions"].astype(str)
    segments = make_segments(labels)
    summary_rows = []
    rollout_tables = []
    split_records = {}

    for state in STATES:
        split = split_segments(segments, state)
        split_records[state] = {name: table.to_dict(orient="records") for name, table in split.items()}
        train_index = np.concatenate([np.arange(row.start, row.stop) for row in split["train"].itertuples(index=False)])
        mean = values[train_index].mean(axis=0)
        std = values[train_index].std(axis=0)
        z = (values - mean) / np.where(std > 0, std, 1.0)
        for history in args.histories:
            for kind in ("persistence", "self", "self_global", "full_var"):
                model, alpha, val_loss = select_and_fit(z, split, history, kind, args.alphas)
                one_step = one_step_metrics(model, z, split["test"])
                rollout = rollout_metrics(model, z, split["test"], args.max_rollout_steps, args.rollout_stride, args.sampling_hz)
                rollout.insert(0, "model", kind)
                rollout.insert(0, "history", history)
                rollout.insert(0, "state", state)
                rollout_tables.append(rollout)
                summary_rows.append({"state": state, "history": history, "model": kind, "alpha": alpha, "validation_mse": val_loss, **one_step, "pearson_gt_0p5_seconds": threshold_horizon(rollout, "pearson_flat", 0.5), "r2_gt_0_seconds": threshold_horizon(rollout, "r2_flat", 0.0), "rollout_cap_seconds": float(rollout.seconds.max()) if len(rollout) else 0.0})

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "model_summary.csv", index=False)
    pd.concat(rollout_tables, ignore_index=True).to_csv(args.output_dir / "rollout_metrics.csv", index=False)
    (args.output_dir / "splits.json").write_text(json.dumps(split_records, indent=2), encoding="utf-8")
    (args.output_dir / "regions.json").write_text(json.dumps(region_names.tolist(), indent=2), encoding="utf-8")
    print(summary.sort_values(["state", "one_step_r2_flat"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
