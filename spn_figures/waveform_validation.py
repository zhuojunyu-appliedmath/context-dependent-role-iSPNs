from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


FSI_HALF_WIDTH_MS = 0.15
FSI_RATE_HZ = 10.0


def select_peak_channel_waveform(
    cluster_waveforms: np.ndarray,
    waveform_channels: np.ndarray,
    peak_channel: int,
) -> tuple[np.ndarray, int, str]:
    """Return a cluster's waveform on its peak channel.

    Parameters
    ----------
    cluster_waveforms
        Array with shape (samples, saved_channels).
    waveform_channels
        Probe-channel identifiers for the saved waveform columns.
    peak_channel
        Probe-channel identifier reported as the cluster's peak channel.

    The exact channel identifier is used whenever it is present.  The
    peak-to-peak maximum is only a guarded fallback, and the caller can retain
    the returned selection label for quality control.
    """
    waveforms = np.asarray(cluster_waveforms, dtype=float)
    channels = np.asarray(waveform_channels).reshape(-1)
    if waveforms.ndim != 2:
        raise ValueError(f"Expected (samples, channels), got {waveforms.shape}.")
    if waveforms.shape[1] != channels.size:
        raise ValueError(
            "Waveform channel metadata do not match the waveform array: "
            f"{waveforms.shape[1]} columns versus {channels.size} identifiers."
        )

    exact = np.flatnonzero(channels.astype(int) == int(peak_channel))
    if exact.size:
        column = int(exact[0])
        selection = "reported_peak_channel"
    else:
        finite_ptp = np.ptp(waveforms, axis=0)
        if not np.isfinite(finite_ptp).any():
            raise ValueError("No finite waveform channel is available.")
        column = int(np.nanargmax(finite_ptp))
        selection = "max_ptp_fallback"
    return waveforms[:, column].copy(), column, selection


def _linear_crossing(y0: float, y1: float, level: float, x0: float) -> float:
    if y1 == y0:
        return float(x0 + 0.5)
    return float(x0 + (level - y0) / (y1 - y0))


def measure_trough_half_width(
    waveform: np.ndarray,
    sample_rate_hz: float = 30_000.0,
    baseline_edge_samples: int = 6,
) -> dict[str, float | int | bool | str]:
    """Measure the negative trough's full width at half amplitude.

    The baseline is the median of samples at the two waveform edges.  The
    nearest half-amplitude crossing on each side of the negative trough is
    linearly interpolated.  Polarity is never silently flipped: a waveform
    whose positive excursion exceeds its negative excursion is retained but
    explicitly flagged for visual review.
    """
    raw = np.asarray(waveform, dtype=float).reshape(-1)
    result: dict[str, float | int | bool | str] = {
        "trough_half_width_ms": np.nan,
        "trough_index": -1,
        "left_crossing_sample": np.nan,
        "right_crossing_sample": np.nan,
        "baseline": np.nan,
        "trough_amplitude": np.nan,
        "positive_amplitude": np.nan,
        "negative_to_positive_ratio": np.nan,
        "trough_to_later_peak_ms": np.nan,
        "width_valid": False,
        "qc_status": "invalid",
        "qc_warning": "",
    }
    if raw.size < 7 or not np.isfinite(raw).all():
        result["qc_status"] = "nonfinite_or_too_short"
        return result
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive and finite.")

    edge = int(min(max(2, baseline_edge_samples), max(2, raw.size // 4)))
    baseline = float(np.median(np.r_[raw[:edge], raw[-edge:]]))
    y = raw - baseline
    trough_index = int(np.argmin(y))
    trough = float(y[trough_index])
    positive = float(np.max(y))
    negative = float(-trough)
    result.update(
        {
            "trough_index": trough_index,
            "baseline": baseline,
            "trough_amplitude": trough,
            "positive_amplitude": positive,
            "negative_to_positive_ratio": negative / max(positive, np.finfo(float).eps),
        }
    )

    if trough_index == 0 or trough_index == raw.size - 1 or trough >= 0:
        result["qc_status"] = "no_interior_negative_trough"
        return result

    half_level = trough / 2.0
    left_candidates = np.flatnonzero(y[:trough_index] >= half_level)
    right_candidates = np.flatnonzero(y[trough_index + 1 :] >= half_level)
    if left_candidates.size == 0 or right_candidates.size == 0:
        result["qc_status"] = "missing_half_amplitude_crossing"
        return result

    left0 = int(left_candidates[-1])
    right1 = int(trough_index + 1 + right_candidates[0])
    left_crossing = _linear_crossing(y[left0], y[left0 + 1], half_level, left0)
    right_crossing = _linear_crossing(y[right1 - 1], y[right1], half_level, right1 - 1)
    width_ms = (right_crossing - left_crossing) * 1000.0 / float(sample_rate_hz)

    after = y[trough_index + 1 :]
    if after.size:
        later_peak_index = trough_index + 1 + int(np.argmax(after))
        trough_to_later_peak_ms = (
            (later_peak_index - trough_index) * 1000.0 / float(sample_rate_hz)
        )
    else:
        trough_to_later_peak_ms = np.nan

    if not np.isfinite(width_ms) or width_ms <= 0:
        result["qc_status"] = "nonpositive_width"
        return result

    result.update(
        {
            "trough_half_width_ms": float(width_ms),
            "left_crossing_sample": float(left_crossing),
            "right_crossing_sample": float(right_crossing),
            "trough_to_later_peak_ms": float(trough_to_later_peak_ms),
            "width_valid": True,
            "qc_status": "ok",
            "qc_warning": "positive_excursion_larger" if positive > negative else "",
        }
    )
    return result


def add_operational_classification(
    table: pd.DataFrame,
    half_width_threshold_ms: float = FSI_HALF_WIDTH_MS,
    rate_threshold_hz: float = FSI_RATE_HZ,
) -> pd.DataFrame:
    """Add a conservative extracellular FSI-like screen to an audit table.

    Missing or invalid widths remain unclassified.  Importantly,
    ``not_strongly_fsi_like`` means only that a unit did not satisfy both
    operational FSI-like criteria; it is not a molecular cell-type label.
    """
    out = table.copy()
    valid = out["width_valid"].fillna(False).astype(bool) & np.isfinite(
        pd.to_numeric(out["session_rate_hz"], errors="coerce")
    )
    width = pd.to_numeric(out["trough_half_width_ms"], errors="coerce")
    rate = pd.to_numeric(out["session_rate_hz"], errors="coerce")

    strongly = pd.Series(pd.NA, index=out.index, dtype="boolean")
    strongly.loc[valid] = (
        (width.loc[valid] < float(half_width_threshold_ms))
        & (rate.loc[valid] > float(rate_threshold_hz))
    )
    out["strongly_fsi_like"] = strongly
    out["not_strongly_fsi_like"] = ~strongly
    narrow = pd.Series(pd.NA, index=out.index, dtype="boolean")
    narrow.loc[valid] = width.loc[valid] < float(half_width_threshold_ms)
    out["narrow_trough_lt_0p15ms"] = narrow
    return out


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion."""
    if total <= 0:
        return np.nan, np.nan
    p = successes / total
    denominator = 1.0 + z**2 / total
    center = (p + z**2 / (2.0 * total)) / denominator
    margin = z * np.sqrt(p * (1.0 - p) / total + z**2 / (4.0 * total**2)) / denominator
    return float(center - margin), float(center + margin)


def audit_summary(table: pd.DataFrame, group_columns: Sequence[str] = ()) -> pd.DataFrame:
    """Summarize coverage and operational FSI-like counts with Wilson CIs."""
    columns = list(group_columns)
    groups = [((), table)] if not columns else table.groupby(columns, dropna=False, sort=False)
    rows: list[dict[str, object]] = []
    for key, group in groups:
        key_tuple = key if isinstance(key, tuple) else (key,)
        valid = group["strongly_fsi_like"].notna()
        n_total = int(len(group))
        n_valid = int(valid.sum())
        n_fsi = int(group.loc[valid, "strongly_fsi_like"].astype(bool).sum())
        n_not = n_valid - n_fsi
        low, high = wilson_interval(n_not, n_valid)
        row: dict[str, object] = dict(zip(columns, key_tuple))
        row.update(
            {
                "n_retained_units": n_total,
                "n_valid_for_joint_screen": n_valid,
                "coverage": n_valid / n_total if n_total else np.nan,
                "n_strongly_fsi_like": n_fsi,
                "n_not_strongly_fsi_like": n_not,
                "fraction_not_strongly_fsi_like": n_not / n_valid if n_valid else np.nan,
                "wilson95_low": low,
                "wilson95_high": high,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)
