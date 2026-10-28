#!/usr/bin/env python3
"""Compare interpretable regional discrete-time equation libraries."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


STATES = ("Awake", "NREM", "REM")
KINDS = ("persistence", "self_linear", "self_cubic", "global_nonlinear", "full_linear", "full_saturating")


def update_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--audit-dir", type=Path, default=Path("logs/audit_data/initial"))
    parser.add_argument("--output-dir", type=Path, default=Path("logs/regional_equation_library/initial"))
    parser.add_argument("--representations", nargs="+", choices=("raw", "causal_highpass"), default=["raw", "causal_highpass"])
    parser.add_argument("--highpass-seconds", type=float, default=16.0)
    parser.add_argument("--sampling-hz", type=float, default=4.146313525439001)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
    parser.add_argument("--max-rollout-steps", type=int, default=80)
    parser.add_argument("--rollout-stride", type=int, default=40)
    parser.add_argument("--divergence-threshold", type=float, default=50.0)


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


def target_indices(table: pd.DataFrame) -> np.ndarray:
    pieces = [np.arange(row.start + 1, row.stop, dtype=np.int64) for row in table.itertuples(index=False) if row.length > 1]
    return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)


def causal_highpass(values: np.ndarray, segments: pd.DataFrame, width: int) -> np.ndarray:
    output = np.empty_like(values, dtype=np.float64)
    for row in segments.itertuples(index=False):
        block = values[row.start : row.stop]
        cumulative = np.vstack([np.zeros((1, block.shape[1])), np.cumsum(block, axis=0)])
        stop = np.arange(1, len(block) + 1)
        start = np.maximum(0, stop - width)
        baseline = (cumulative[stop] - cumulative[start]) / (stop - start)[:, None]
        output[row.start : row.stop] = block - baseline
    return output


def feature_names(kind: str, region_names: np.ndarray, target: int) -> list[str]:
    own = str(region_names[target])
    if kind == "self_linear":
        return [f"x[{own}]"]
    if kind == "self_cubic":
        return [f"x[{own}]", f"x[{own}]^2", f"x[{own}]^3"]
    if kind == "global_nonlinear":
        return [f"x[{own}]", f"x[{own}]^2", f"x[{own}]^3", "mean(x)", "mean(x)^2", "mean(x)^3", f"x[{own}]*mean(x)"]
    if kind == "full_linear":
        return [f"x[{name}]" for name in region_names]
    if kind == "full_saturating":
        return [f"x[{name}]" for name in region_names] + [f"tanh(x[{name}])" for name in region_names]
    return []


def features(x: np.ndarray, kind: str, target: int) -> np.ndarray:
    own = x[:, target : target + 1]
    if kind == "self_linear":
        return own
    if kind == "self_cubic":
        return np.concatenate([own, own**2, own**3], axis=1)
    if kind == "global_nonlinear":
        mean = x.mean(axis=1, keepdims=True)
        return np.concatenate([own, own**2, own**3, mean, mean**2, mean**3, own * mean], axis=1)
    if kind == "full_linear":
        return x
    if kind == "full_saturating":
        return np.concatenate([x, np.tanh(x)], axis=1)
    raise ValueError(kind)


@dataclass
class EquationModel:
    kind: str
    scalers: list[StandardScaler]
    estimators: list[Ridge]

    def predict_dx(self, x: np.ndarray) -> np.ndarray:
        if self.kind == "persistence":
            return np.zeros_like(x)
        output = np.empty_like(x, dtype=np.float64)
        for target, (scaler, estimator) in enumerate(zip(self.scalers, self.estimators, strict=True)):
            output[:, target] = estimator.predict(scaler.transform(features(x, self.kind, target)))
        return output


def select_and_fit(z: np.ndarray, split: dict[str, pd.DataFrame], kind: str, alphas: list[float]) -> tuple[EquationModel, float, float]:
    if kind == "persistence":
        return EquationModel(kind, [], []), 0.0, float(np.mean(np.diff(z[table_indices(split["validation"])], axis=0) ** 2))
    train_targets = target_indices(split["train"])
    validation_targets = target_indices(split["validation"])
    x_train, y_train = z[train_targets - 1], z[train_targets] - z[train_targets - 1]
    x_validation, y_validation = z[validation_targets - 1], z[validation_targets] - z[validation_targets - 1]
    scalers = [StandardScaler().fit(features(x_train, kind, target)) for target in range(z.shape[1])]
    best_alpha, best_loss = None, np.inf
    for alpha in alphas:
        prediction = np.empty_like(y_validation)
        for target, scaler in enumerate(scalers):
            estimator = Ridge(alpha=alpha).fit(scaler.transform(features(x_train, kind, target)), y_train[:, target])
            prediction[:, target] = estimator.predict(scaler.transform(features(x_validation, kind, target)))
        loss = float(np.mean((prediction - y_validation) ** 2))
        if loss < best_loss:
            best_alpha, best_loss = alpha, loss
    estimators = []
    for target, scaler in enumerate(scalers):
        estimator = Ridge(alpha=best_alpha).fit(scaler.transform(features(x_train, kind, target)), y_train[:, target])
        estimators.append(estimator)
    return EquationModel(kind, scalers, estimators), float(best_alpha), best_loss


def one_step_metrics(model: EquationModel, z: np.ndarray, table: pd.DataFrame) -> dict[str, float]:
    targets = target_indices(table)
    current, truth = z[targets - 1], z[targets]
    dx_truth = truth - current
    dx_prediction = model.predict_dx(current)
    prediction = current + dx_prediction
    node_r2 = [r2_score(dx_truth[:, node], dx_prediction[:, node]) for node in range(z.shape[1])]
    return {"level_r2_flat": float(r2_score(truth.ravel(), prediction.ravel())), "dx_r2_flat": float(r2_score(dx_truth.ravel(), dx_prediction.ravel())), "dx_r2_region_median": float(np.median(node_r2)), "dx_corr_flat": float(np.corrcoef(dx_truth.ravel(), dx_prediction.ravel())[0, 1]) if np.std(dx_prediction) else np.nan, "n_test_targets": len(targets)}


def rollout_metrics(model: EquationModel, z: np.ndarray, table: pd.DataFrame, max_steps: int, stride: int, sampling_hz: float, threshold: float) -> pd.DataFrame:
    records: dict[int, dict[str, list[np.ndarray] | int]] = {}
    for segment in table.itertuples(index=False):
        for origin in range(segment.start, segment.stop - 1, stride):
            simulated = z[origin].copy()
            available = min(max_steps, segment.stop - origin - 1)
            alive = True
            for step in range(1, available + 1):
                if alive:
                    simulated = simulated + model.predict_dx(simulated[None])[0]
                    alive = bool(np.all(np.isfinite(simulated)) and np.max(np.abs(simulated)) < threshold)
                bucket = records.setdefault(step, {"truth": [], "prediction": [], "attempted": 0, "bounded": 0})
                bucket["attempted"] += 1
                if alive:
                    bucket["truth"].append(z[origin + step])
                    bucket["prediction"].append(simulated.copy())
                    bucket["bounded"] += 1
    rows = []
    for step, bucket in sorted(records.items()):
        if bucket["bounded"] > 1:
            truth, prediction = np.stack(bucket["truth"]), np.stack(bucket["prediction"])
            r2 = float(r2_score(truth.ravel(), prediction.ravel()))
            corr = float(np.corrcoef(truth.ravel(), prediction.ravel())[0, 1])
        else:
            r2, corr = np.nan, np.nan
        rows.append({"step": step, "seconds": step / sampling_hz, "r2_flat": r2, "pearson_flat": corr, "bounded_fraction": bucket["bounded"] / bucket["attempted"], "n_origins": bucket["attempted"]})
    return pd.DataFrame(rows)


def coefficient_rows(model: EquationModel, representation: str, state: str, alpha: float, region_names: np.ndarray) -> list[dict]:
    rows = []
    for target, (scaler, estimator) in enumerate(zip(model.scalers, model.estimators, strict=True)):
        raw_coef = estimator.coef_ / scaler.scale_
        intercept = estimator.intercept_ - np.dot(estimator.coef_, scaler.mean_ / scaler.scale_)
        rows.append({"representation": representation, "state": state, "model": model.kind, "target": str(region_names[target]), "term": "intercept", "coefficient_per_frame": float(intercept), "alpha": alpha})
        for name, coefficient in zip(feature_names(model.kind, region_names, target), raw_coef, strict=True):
            rows.append({"representation": representation, "state": state, "model": model.kind, "target": str(region_names[target]), "term": name, "coefficient_per_frame": float(coefficient), "alpha": alpha})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    update_parser(parser)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = np.load(args.audit_dir / "region_means.npz", allow_pickle=True)
    raw = archive["values"].astype(np.float64)
    labels = archive["labels"].astype(str)
    region_names = archive["regions"].astype(str)
    segments = make_segments(labels)
    width = max(3, int(round(args.highpass_seconds * args.sampling_hz)))
    representations = {"raw": raw, "causal_highpass": causal_highpass(raw, segments, width)}
    summary_rows, rollout_tables, coefficients = [], [], []
    split_records = {}
    for state in STATES:
        split = split_segments(segments, state)
        split_records[state] = {name: table.to_dict(orient="records") for name, table in split.items()}
        for representation in args.representations:
            values = representations[representation]
            train_index = table_indices(split["train"])
            mean, std = values[train_index].mean(axis=0), values[train_index].std(axis=0)
            z = (values - mean) / np.where(std > 0, std, 1.0)
            for kind in KINDS:
                model, alpha, validation_loss = select_and_fit(z, split, kind, args.alphas)
                metrics = one_step_metrics(model, z, split["test"])
                rollout = rollout_metrics(model, z, split["test"], args.max_rollout_steps, args.rollout_stride, args.sampling_hz, args.divergence_threshold)
                rollout.insert(0, "model", kind)
                rollout.insert(0, "state", state)
                rollout.insert(0, "representation", representation)
                rollout_tables.append(rollout)
                at_four_seconds = rollout.iloc[(rollout.seconds - 4.0).abs().argmin()]
                summary_rows.append({"representation": representation, "state": state, "model": kind, "alpha": alpha, "validation_mse": validation_loss, **metrics, "rollout_r2_at_4s": at_four_seconds.r2_flat, "rollout_corr_at_4s": at_four_seconds.pearson_flat, "bounded_fraction_at_4s": at_four_seconds.bounded_fraction})
                coefficients.extend(coefficient_rows(model, representation, state, alpha, region_names))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "model_summary.csv", index=False)
    pd.concat(rollout_tables, ignore_index=True).to_csv(args.output_dir / "rollout_metrics.csv", index=False)
    pd.DataFrame(coefficients).to_csv(args.output_dir / "coefficients.csv", index=False)
    (args.output_dir / "splits.json").write_text(json.dumps(split_records, indent=2), encoding="utf-8")
    print(summary.sort_values(["representation", "state", "dx_r2_flat"], ascending=[True, True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
