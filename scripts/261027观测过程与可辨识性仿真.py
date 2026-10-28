#!/usr/bin/env python3
"""Stress-test recovery after candidate low-pass observation and sampling models."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d


def update_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", type=Path, default=Path("logs/observation_identifiability/initial"))
    parser.add_argument("--duration-seconds", type=float, default=900.0)
    parser.add_argument("--latent-hz", type=float, default=100.0)
    parser.add_argument("--imaging-hz", type=float, default=4.146313525439001)
    parser.add_argument("--exposure-seconds", type=float, default=0.1)
    parser.add_argument("--n-seeds", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261009)


def lowpass(signal: np.ndarray, tau: float, dt: float) -> np.ndarray:
    if tau <= 0:
        return signal.copy()
    output = np.empty_like(signal)
    output[0] = signal[0]
    alpha = 1.0 - np.exp(-dt / tau)
    for index in range(1, len(signal)):
        output[index] = output[index - 1] + alpha * (signal[index] - output[index - 1])
    return output


def observe(signal: np.ndarray, calcium_tau: float, exposure_seconds: float, latent_hz: float, imaging_hz: float) -> np.ndarray:
    filtered = lowpass(signal, calcium_tau, 1.0 / latent_hz)
    width = max(1, int(round(exposure_seconds * latent_hz)))
    integrated = uniform_filter1d(filtered, size=width, axis=0, mode="nearest") if width > 1 else filtered
    sample_index = np.round(np.arange(1.0 / imaging_hz, len(signal) / latent_hz, 1.0 / imaging_hz) * latent_hz).astype(int) - 1
    return integrated[np.clip(sample_index, 0, len(integrated) - 1)]


def simulate_ou(tau: float, n_step: int, dt: float, rng: np.random.Generator) -> np.ndarray:
    coefficient = np.exp(-dt / tau)
    innovation = np.sqrt(max(1.0 - coefficient**2, 0.0))
    output = np.empty(n_step)
    output[0] = rng.normal()
    noise = rng.normal(size=n_step - 1)
    for index in range(1, n_step):
        output[index] = coefficient * output[index - 1] + innovation * noise[index - 1]
    return output


def apparent_timescale(signal: np.ndarray, sampling_hz: float) -> tuple[float, float]:
    left = signal[:-1] - signal[:-1].mean()
    right = signal[1:] - signal[1:].mean()
    coefficient = float(np.dot(left, right) / np.dot(left, left))
    clipped = np.clip(coefficient, 1e-6, 1 - 1e-9)
    return coefficient, float(-1.0 / sampling_hz / np.log(clipped))


def timescale_experiment(args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    n_step = int(round(args.duration_seconds * args.latent_hz))
    dt = 1.0 / args.latent_hz
    tau_neural_values = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
    calcium_values = (0.0, 0.2, 0.5, 1.0)
    for seed_offset in range(args.n_seeds):
        rng = np.random.default_rng(args.seed + seed_offset)
        for tau_neural in tau_neural_values:
            latent = simulate_ou(tau_neural, n_step, dt, rng)
            for tau_calcium in calcium_values:
                for exposure in (0.0, args.exposure_seconds):
                    measured = observe(latent, tau_calcium, exposure, args.latent_hz, args.imaging_hz)
                    ar1, estimate = apparent_timescale(measured, args.imaging_hz)
                    rows.append({"seed": args.seed + seed_offset, "tau_neural_seconds": tau_neural, "tau_calcium_seconds": tau_calcium, "exposure_seconds": exposure, "ar1": ar1, "apparent_tau_seconds": estimate})
    return pd.DataFrame(rows)


def simulate_delay_pair(delay_seconds: float, n_step: int, dt: float, rng: np.random.Generator) -> np.ndarray:
    source = lowpass(rng.normal(size=n_step), 0.05, dt)
    delay = int(round(delay_seconds / dt))
    shifted = np.zeros(n_step)
    if delay:
        shifted[delay:] = source[:-delay]
    else:
        shifted[:] = source
    target = lowpass(shifted + 0.7 * rng.normal(size=n_step), 0.08, dt)
    return np.column_stack([source, target])


def peak_delay(signal: np.ndarray, sampling_hz: float, max_lag_steps: int = 8) -> tuple[int, float, float]:
    source = signal[:, 0]
    target = signal[:, 1]
    correlations = []
    for lag in range(max_lag_steps + 1):
        left = source[: len(source) - lag or None]
        right = target[lag:]
        correlations.append(float(np.corrcoef(left, right)[0, 1]))
    peak = int(np.nanargmax(correlations))
    return peak, peak / sampling_hz, correlations[peak]


def delay_experiment(args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    n_step = int(round(args.duration_seconds * args.latent_hz))
    delays = (0.0, 0.025, 0.05, 0.1, 0.2, 0.4, 0.75, 1.0)
    calcium_values = (0.0, 0.2, 0.5, 1.0)
    for seed_offset in range(args.n_seeds):
        rng = np.random.default_rng(args.seed + 1000 + seed_offset)
        for true_delay in delays:
            latent = simulate_delay_pair(true_delay, n_step, 1.0 / args.latent_hz, rng)
            latent_peak, latent_estimate, latent_corr = peak_delay(latent, args.latent_hz, int(2 * args.latent_hz))
            rows.append({"seed": args.seed + seed_offset, "true_delay_seconds": true_delay, "tau_calcium_seconds": 0.0, "exposure_seconds": 0.0, "sampling_hz": args.latent_hz, "estimated_delay_steps": latent_peak, "estimated_delay_seconds": latent_estimate, "peak_correlation": latent_corr})
            for tau_calcium in calcium_values:
                measured = observe(latent, tau_calcium, args.exposure_seconds, args.latent_hz, args.imaging_hz)
                peak, estimate, corr = peak_delay(measured, args.imaging_hz)
                rows.append({"seed": args.seed + seed_offset, "true_delay_seconds": true_delay, "tau_calcium_seconds": tau_calcium, "exposure_seconds": args.exposure_seconds, "sampling_hz": args.imaging_hz, "estimated_delay_steps": peak, "estimated_delay_seconds": estimate, "peak_correlation": corr})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    update_parser(parser)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timescale_experiment(args).to_csv(args.output_dir / "timescale_recovery.csv", index=False)
    delay_experiment(args).to_csv(args.output_dir / "delay_recovery.csv", index=False)


if __name__ == "__main__":
    main()
