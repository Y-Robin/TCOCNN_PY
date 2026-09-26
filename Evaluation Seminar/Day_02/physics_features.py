"""Physically motivated features from one raw MOX signal."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dataset_pipeline import (
    CYCLE_SAMPLES,
    HIGH_PHASE_SAMPLES,
    HIGH_TEMPERATURE_C,
    LOW_PHASE_SAMPLES,
    LOW_TEMPERATURES_C,
    SAMPLE_RATE_HZ,
)


@dataclass(frozen=True)
class TemperaturePhase:
    start: int
    stop: int
    temperature_c: int
    kind: str
    cycle_step: int

    @property
    def label(self) -> str:
        return f"step{self.cycle_step:02d}_{self.kind}_t{self.temperature_c:03d}"


def temperature_phases() -> tuple[TemperaturePhase, ...]:
    """The 24 constant-temperature phases of one cycle."""
    result: list[TemperaturePhase] = []
    block = HIGH_PHASE_SAMPLES + LOW_PHASE_SAMPLES
    for step, low_temperature in enumerate(LOW_TEMPERATURES_C):
        base = step * block
        result.append(
            TemperaturePhase(
                base,
                base + HIGH_PHASE_SAMPLES,
                HIGH_TEMPERATURE_C,
                "high",
                step,
            )
        )
        result.append(
            TemperaturePhase(
                base + HIGH_PHASE_SAMPLES,
                base + block,
                low_temperature,
                "low",
                step,
            )
        )
    return tuple(result)


def _validate_cycles(X: np.ndarray) -> np.ndarray:
    values = np.asarray(X, dtype=float)
    if values.ndim != 3 or values.shape[1:] != (1, CYCLE_SAMPLES):
        raise ValueError(
            f"Expected (n, 1, {CYCLE_SAMPLES}), received: {values.shape}"
        )
    return values[:, 0, :]


def _initial_slope(values: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    width = min(10, values.shape[1])
    t = np.arange(width, dtype=float) / sample_rate_hz
    centered = t - t.mean()
    return (values[:, :width] @ centered) / np.sum(centered**2)


def _tau63(
    values: np.ndarray,
    start_level: np.ndarray,
    end_level: np.ndarray,
    sample_rate_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Zeit bis 63,2 % des beobachteten Phasenhubs, linear interpoliert."""
    amplitude = end_level - start_level
    scale = np.maximum(np.ptp(values, axis=1), 1.0)
    valid_amplitude = np.abs(amplitude) > 1e-5 * scale
    safe_amplitude = np.where(valid_amplitude, amplitude, 1.0)
    progress = (values - start_level[:, None]) / safe_amplitude[:, None]
    crossing = progress >= (1.0 - np.exp(-1.0))
    # t=0 ist per Definition noch keine gemessene Zeitkonstante.
    crossing[:, 0] = False
    found = crossing.any(axis=1) & valid_amplitude
    index = crossing.argmax(axis=1)

    lower_index = np.maximum(index - 1, 0)
    row = np.arange(len(values))
    lower = progress[row, lower_index]
    upper = progress[row, index]
    denominator = upper - lower
    fraction = np.divide(
        (1.0 - np.exp(-1.0)) - lower,
        denominator,
        out=np.zeros_like(lower),
        where=np.abs(denominator) > 1e-12,
    )
    sample_position = lower_index + np.clip(fraction, 0.0, 1.0)
    duration = (values.shape[1] - 1) / sample_rate_hz
    tau = np.where(found, sample_position / sample_rate_hz, duration)
    return tau, found.astype(float)


def extract_phase_features(
    X: np.ndarray,
    sample_rate_hz: float = SAMPLE_RATE_HZ,
) -> tuple[np.ndarray, list[str]]:
    """Sechs dynamische Merkmale je Hoch-/Niedrigtemperaturphase.

    tau63 is the time to 63.2% of the observed signal amplitude. For an
    ideal first-order response, it equals the time constant tau.
    """
    cycles = _validate_cycles(X)
    columns: list[np.ndarray] = []
    names: list[str] = []

    for phase in temperature_phases():
        values = cycles[:, phase.start : phase.stop]
        end_width = max(3, int(round(0.15 * values.shape[1])))
        # A short mean makes the starting level more robust to individual noise.
        start_width = min(3, values.shape[1])
        start_level = values[:, :start_width].mean(axis=1)
        end_level = values[:, -end_width:].mean(axis=1)
        amplitude = end_level - start_level
        tau, crossing_found = _tau63(
            values, start_level, end_level, sample_rate_hz
        )
        initial_slope = _initial_slope(values, sample_rate_hz)

        time = np.arange(values.shape[1], dtype=float) / sample_rate_hz
        safe_tau = np.maximum(tau, 1.0 / sample_rate_hz)
        exponential = start_level[:, None] + amplitude[:, None] * (
            1.0 - np.exp(-time[None, :] / safe_tau[:, None])
        )
        fit_rmse = np.sqrt(np.mean((values - exponential) ** 2, axis=1))

        for suffix, feature in (
            ("start", start_level),
            ("end", end_level),
            ("amplitude", amplitude),
            ("tau63_s", tau),
            ("initial_slope_per_s", initial_slope),
            ("exp_fit_rmse", fit_rmse),
            ("tau_crossing_found", crossing_found),
        ):
            columns.append(feature)
            names.append(f"{phase.label}__{suffix}")

    return np.column_stack(columns), names


def learn_ala_breakpoints(
    reference_signal: np.ndarray,
    n_segments: int = 50,
    min_segment_length: int = 8,
) -> np.ndarray:
    """Learns adaptive breakpoints from maximum reconstruction error."""
    reference = np.asarray(reference_signal, dtype=float).reshape(-1)
    if len(reference) != CYCLE_SAMPLES:
        raise ValueError(f"Reference signal must contain {CYCLE_SAMPLES} values")
    if not 2 <= n_segments <= len(reference) // min_segment_length:
        raise ValueError("Invalid number of ALA segments")

    x = np.arange(len(reference))
    breakpoints = [0, len(reference) - 1]
    while len(breakpoints) - 1 < n_segments:
        ordered = np.array(sorted(breakpoints))
        reconstruction = np.interp(x, ordered, reference[ordered])
        score = (reference - reconstruction) ** 2
        allowed = np.zeros(len(reference), dtype=bool)
        for left, right in zip(ordered[:-1], ordered[1:]):
            lo = left + min_segment_length
            hi = right - min_segment_length
            if lo <= hi:
                allowed[lo : hi + 1] = True
        if not allowed.any():
            raise RuntimeError("Requested segment count cannot be reached")
        score[~allowed] = -np.inf
        breakpoints.append(int(np.argmax(score)))
    return np.array(sorted(breakpoints), dtype=int)


def extract_ala_features(
    X: np.ndarray,
    breakpoints: np.ndarray,
    sample_rate_hz: float = SAMPLE_RATE_HZ,
) -> tuple[np.ndarray, list[str]]:
    """Mean and linear slope for every fixed ALA segment."""
    cycles = _validate_cycles(X)
    points = np.asarray(breakpoints, dtype=int)
    if points[0] != 0 or points[-1] != CYCLE_SAMPLES - 1:
        raise ValueError("ALA breakpoints must cover the complete cycle")
    if np.any(np.diff(points) <= 0):
        raise ValueError("ALA breakpoints must be strictly increasing")

    columns: list[np.ndarray] = []
    names: list[str] = []
    for segment, (left, right) in enumerate(zip(points[:-1], points[1:])):
        values = cycles[:, left : right + 1]
        t = np.arange(values.shape[1], dtype=float) / sample_rate_hz
        centered = t - t.mean()
        slope = (values @ centered) / np.sum(centered**2)
        columns.extend((values.mean(axis=1), slope))
        names.extend(
            (
                f"ala_segment{segment:02d}__mean",
                f"ala_segment{segment:02d}__slope_per_s",
            )
        )
    return np.column_stack(columns), names


def reconstruct_ala(signal: np.ndarray, breakpoints: np.ndarray) -> np.ndarray:
    """Reconstruction with a separate least-squares line per segment."""
    values = np.asarray(signal, dtype=float).reshape(-1)
    result = np.empty_like(values)
    for left, right in zip(breakpoints[:-1], breakpoints[1:]):
        x = np.arange(left, right + 1, dtype=float)
        design = np.column_stack((x, np.ones(len(x))))
        coefficients = np.linalg.lstsq(design, values[left : right + 1], rcond=None)[0]
        result[left : right + 1] = design @ coefficients
    return result
