from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, confusion_matrix

from .config import LAST_K, SPN_NAMES, STAGE_B_WINDOW_S


def zscore_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-9)


def stage_indices(
    bin_centers_s: np.ndarray,
    last_k: int = LAST_K,
    premove_gap_s: float = 0.010,
    stage_b_window_s: tuple[float, float] = STAGE_B_WINDOW_S,
) -> tuple[np.ndarray, np.ndarray]:
    centers = np.asarray(bin_centers_s, dtype=float)
    bin_size = float(np.median(np.diff(centers)))
    right_edges = centers + bin_size / 2.0
    stage_a = np.flatnonzero(right_edges <= -premove_gap_s + 1e-12)[-last_k:]
    stage_b = np.flatnonzero((centers >= stage_b_window_s[0]) & (centers < stage_b_window_s[1]))
    return stage_a, stage_b


def make_12d_features(
    rates: np.ndarray,
    left_trials: np.ndarray,
    right_trials: np.ndarray,
    stage_a_bins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left_mean = rates[:, left_trials][:, :, stage_a_bins].mean(axis=1)
    right_mean = rates[:, right_trials][:, :, stage_a_bins].mean(axis=1)
    x12 = np.concatenate([left_mean, right_mean], axis=1)
    return x12, left_mean, right_mean


def empirical_unit_filter(
    x12: np.ndarray,
    left_mean: np.ndarray,
    right_mean: np.ndarray,
    time_s: np.ndarray | None = None,
    min_profile_range: float = 0.05,
    min_slope: float = 0.0,
) -> np.ndarray:
    x = (
        np.arange(left_mean.shape[1], dtype=float)
        if time_s is None
        else np.asarray(time_s, dtype=float)
    )
    slope_left = np.array([np.polyfit(x, row, 1)[0] for row in left_mean])
    slope_right = np.array([np.polyfit(x, row, 1)[0] for row in right_mean])
    return (
        (np.ptp(x12, axis=1) >= min_profile_range)
        & (slope_left >= min_slope)
        & (slope_right >= min_slope)
    )

def stage1_channel_labels(x12: np.ndarray) -> tuple[np.ndarray, int]:
    x12_z = zscore_rows(x12)
    raw = AgglomerativeClustering(n_clusters=2, linkage="ward", metric="euclidean").fit_predict(x12_z)
    half = x12.shape[1] // 2
    bias = x12[:, :half].mean(axis=1) - x12[:, half:].mean(axis=1)
    left_cluster = int(np.argmax([bias[raw == k].mean() for k in (0, 1)]))
    channel = np.where(raw == left_cluster, 0, 1)
    return channel, left_cluster


def correlation_subclusters(
    rates: np.ndarray,
    unit_indices: np.ndarray,
    trial_indices: np.ndarray,
    stage_b_bins: np.ndarray,
) -> np.ndarray:
    vectors = rates[unit_indices][:, trial_indices][:, :, stage_b_bins].reshape(len(unit_indices), -1)
    vectors = zscore_rows(vectors)
    corr = np.corrcoef(vectors)
    distance = 1.0 - np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    return AgglomerativeClustering(n_clusters=2, linkage="average", metric="precomputed").fit_predict(distance)



def correlation_subclusters_pruned(
    rates: np.ndarray,
    unit_indices: np.ndarray,
    trial_indices: np.ndarray,
    stage_b_bins: np.ndarray,
    minimum_subcluster_size: int = 3,
    max_iterations: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster one channel and remove undersized subclusters."""
    vectors = rates[unit_indices][:, trial_indices][:, :, stage_b_bins].reshape(
        len(unit_indices), -1
    )
    finite = np.all(np.isfinite(vectors), axis=1) & (vectors.std(axis=1) >= 1e-6)
    kept = np.flatnonzero(finite)

    for _ in range(max_iterations):
        if len(kept) < 2 * minimum_subcluster_size:
            break
        x = zscore_rows(vectors[kept])
        corr = np.nan_to_num(np.clip(np.corrcoef(x), -1.0, 1.0), nan=0.0)
        np.fill_diagonal(corr, 1.0)
        distance = 1.0 - corr
        np.fill_diagonal(distance, 0.0)
        labels = AgglomerativeClustering(
            n_clusters=2,
            linkage="average",
            metric="precomputed",
        ).fit_predict(distance)
        counts = np.bincount(labels, minlength=2)
        if counts.min() >= minimum_subcluster_size:
            return kept, labels
        kept = kept[labels != int(np.argmin(counts))]

    raise ValueError(
        "Too few units for two stable Stage 2 subclusters after outlier pruning."
    )

def split_fast_slow(decision_time_ms: np.ndarray, trial_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = trial_indices[np.argsort(decision_time_ms[trial_indices])]
    cut = len(ordered) // 2
    return ordered[:cut], ordered[cut:]


def cluster_mean_vectors(
    rates: np.ndarray,
    final_labels: np.ndarray,
    trial_indices: np.ndarray,
    stage_a_bins: np.ndarray,
) -> list[np.ndarray]:
    vectors: list[np.ndarray] = []
    for label in range(4):
        units = np.flatnonzero(final_labels == label)
        trial_by_bin = rates[units][:, trial_indices][:, :, stage_a_bins].mean(axis=0)
        vector = trial_by_bin.reshape(-1)
        vectors.append((vector - vector.mean()) / (vector.std() + 1e-9))
    return vectors


def correlation_matrix(vectors: list[np.ndarray]) -> np.ndarray:
    return np.corrcoef(np.vstack(vectors))


def infer_pathway_names(
    rates: np.ndarray,
    final_labels: np.ndarray,
    decision_time_ms: np.ndarray,
    trials: np.ndarray,
    stage_a_bins: np.ndarray,
) -> tuple[np.ndarray, dict[int, str], np.ndarray, np.ndarray, float]:
    fast, slow = split_fast_slow(decision_time_ms, trials)
    corr_fast = correlation_matrix(cluster_mean_vectors(rates, final_labels, fast, stage_a_bins))
    corr_slow = correlation_matrix(cluster_mean_vectors(rates, final_labels, slow, stage_a_bins))
    delta = corr_slow - corr_fast
    candidates = [(left, right, delta[left, right]) for left in (0, 1) for right in (2, 3)]
    d_left, d_right, best_drop = min(candidates, key=lambda item: item[2])
    mapping = {
        int(d_left): "dSPN_left",
        int(1 - d_left): "iSPN_left",
        int(d_right): "dSPN_right",
        int(5 - d_right): "iSPN_right",
    }
    names = np.array([mapping[int(label)] for label in final_labels], dtype=object)
    return names, mapping, corr_fast, corr_slow, float(best_drop)


def cluster_empirical_session(payload: dict[str, Any]) -> dict[str, Any]:
    rates = np.asarray(payload["rates"], dtype=float)
    choice_left = np.asarray(payload["choice_left"], dtype=bool)
    decision_time_ms = np.asarray(payload["decision_time_ms"], dtype=float)
    unit_ids = np.asarray(payload["unit_ids"], dtype=int)
    trial_ids = np.asarray(payload["trial_ids"], dtype=int)
    bin_centers_s = np.asarray(payload["bin_centers_s"], dtype=float)

    left_trials = np.flatnonzero(choice_left)
    right_trials = np.flatnonzero(~choice_left)
    stage_a, stage_b = stage_indices(
        bin_centers_s,
        premove_gap_s=float(payload.get("premove_gap_s", 0.010)),
    )
    x12, left_mean, right_mean = make_12d_features(
        rates, left_trials, right_trials, stage_a
    )
    keep = empirical_unit_filter(
        x12,
        left_mean,
        right_mean,
        time_s=bin_centers_s[stage_a],
        min_profile_range=float(payload.get("min_profile_range", 0.05)),
        min_slope=float(payload.get("ramp_min_slope", 0.0)),
    )

    rates = rates[keep]
    unit_ids = unit_ids[keep]
    x12 = x12[keep]
    x12_z = zscore_rows(x12)
    minimum_good_units = int(payload.get("minimum_good_units", 1))
    if len(unit_ids) < minimum_good_units:
        raise ValueError(
            f"{payload['session']}: only {len(unit_ids)} units remain after filtering."
        )

    channel_labels, _ = stage1_channel_labels(x12)
    channel_counts = np.bincount(channel_labels, minlength=2)
    if channel_counts.min() < 2:
        raise ValueError(f"{payload['session']}: Stage 1 produced an undersized channel.")

    final_labels = np.full(len(unit_ids), -1, dtype=int)
    all_trials = np.sort(np.concatenate([left_trials, right_trials]))
    minimum_subcluster_size = int(payload.get("minimum_subcluster_size", 1))
    for channel, offset in ((0, 0), (1, 2)):
        unit_idx = np.flatnonzero(channel_labels == channel)
        if minimum_subcluster_size > 1:
            kept_local, sublabels = correlation_subclusters_pruned(
                rates,
                unit_idx,
                all_trials,
                stage_b,
                minimum_subcluster_size=minimum_subcluster_size,
            )
            final_labels[unit_idx[kept_local]] = sublabels + offset
        else:
            final_labels[unit_idx] = correlation_subclusters(
                rates, unit_idx, all_trials, stage_b
            ) + offset

    assigned = final_labels >= 0
    rates = rates[assigned]
    unit_ids = unit_ids[assigned]
    x12 = x12[assigned]
    x12_z = x12_z[assigned]
    channel_labels = channel_labels[assigned]
    final_labels = final_labels[assigned]
    if set(np.unique(final_labels)) != {0, 1, 2, 3}:
        raise ValueError(f"{payload['session']}: all four SPN subclusters were not recovered.")

    final_names, name_map, corr_fast, corr_slow, corr_drop = infer_pathway_names(
        rates, final_labels, decision_time_ms, all_trials, stage_a
    )

    return {
        **payload,
        "rates": rates,
        "unit_ids": unit_ids,
        "stage_a_bins": stage_a,
        "stage_b_bins": stage_b,
        "x12": x12,
        "x12_z": x12_z,
        "channel_labels": channel_labels,
        "final_labels": final_labels,
        "final_names": final_names,
        "name_map": name_map,
        "corr_fast": corr_fast,
        "corr_slow": corr_slow,
        "corr_drop": corr_drop,
        "trial_ids": trial_ids,
    }


def attach_unit_labels(
    payload: dict[str, Any],
    reference_unit_ids: np.ndarray,
    reference_names: np.ndarray,
) -> dict[str, Any]:
    name_by_unit = {
        int(unit_id): str(name)
        for unit_id, name in zip(reference_unit_ids, reference_names)
    }
    unit_ids = np.asarray(payload["unit_ids"], dtype=int)
    keep = np.array([int(unit_id) in name_by_unit for unit_id in unit_ids])
    names = np.array([name_by_unit[int(unit_id)] for unit_id in unit_ids[keep]], dtype=object)
    label_by_name = {
        "dSPN_left": 0,
        "iSPN_left": 1,
        "dSPN_right": 2,
        "iSPN_right": 3,
    }
    return {
        **payload,
        "rates": np.asarray(payload["rates"])[keep],
        "unit_ids": unit_ids[keep],
        "final_names": names,
        "final_labels": np.array([label_by_name[name] for name in names], dtype=int),
    }


def _align_cluster_ids(true_labels: np.ndarray, raw_labels: np.ndarray) -> np.ndarray:
    """Map arbitrary cluster IDs to the known model label order."""
    matrix = confusion_matrix(true_labels, raw_labels)
    rows, cols = linear_sum_assignment(matrix.max() - matrix)
    mapping = {int(cluster): int(label) for label, cluster in zip(rows, cols)}
    return np.array([mapping[int(label)] for label in raw_labels], dtype=int)


def _choice_matched_selectivity(x12_z: np.ndarray, row: int, channel: int) -> float:
    """Return the choice-matched 12D contrast used only to break exact ties."""
    left = float(x12_z[row, :6].mean())
    right = float(x12_z[row, 6:].mean())
    return left - right if channel == 0 else right - left


def _direct_candidate(
    cross_channel_scores: np.ndarray,
    rows: np.ndarray,
    channel: int,
    x12_z: np.ndarray,
) -> int:
    """Return the local candidate assigned to dSPN within one action channel."""
    if not np.isclose(cross_channel_scores[0], cross_channel_scores[1], atol=1e-12, rtol=0.0):
        return int(np.argmin(cross_channel_scores))
    selectivity = [_choice_matched_selectivity(x12_z, int(row), channel) for row in rows]
    return int(np.argmax(selectivity))


def cbgt_ground_truth_clustering(
    feature_table: pd.DataFrame,
    vector_table: pd.DataFrame,
) -> dict[str, Any]:
    """Infer CBGT channel and pathway labels and compare them with model truth.

    Stage 1 applies Ward clustering to the row-z-scored 12D firing-rate
    profiles. Stage 2 is evaluated within each simulated network: among the two
    candidates assigned to each channel, the unit with weaker mean correlation
    to the opposite channel is labeled dSPN and the other is labeled iSPN.
    Pathway ground truth is used only for the final validation metrics.
    """
    feature_cols = [f"feature_{i:02d}" for i in range(12)]
    x12_z = zscore_rows(feature_table[feature_cols].to_numpy(float))
    network_ids = feature_table["network_id"].to_numpy(int)
    true_names = feature_table["true_name"].to_numpy(dtype=str)

    true_channel = np.where(np.char.endswith(true_names, "_left"), 0, 1)
    raw_channel = AgglomerativeClustering(
        n_clusters=2,
        linkage="ward",
        metric="euclidean",
    ).fit_predict(x12_z)
    stage1_labels = _align_cluster_ids(true_channel, raw_channel)

    # Stage 2 requires the known model composition of two SPN populations per channel.
    stage2_channel_labels = stage1_labels.copy()
    centroids = np.vstack(
        [x12_z[stage1_labels == channel].mean(axis=0) for channel in (0, 1)]
    )
    repaired_networks: list[int] = []
    for network_id in np.unique(network_ids):
        rows = np.flatnonzero(network_ids == network_id)
        if np.array_equal(
            np.bincount(stage2_channel_labels[rows], minlength=2),
            np.array([2, 2]),
        ):
            continue

        distances = np.column_stack(
            [np.linalg.norm(x12_z[rows] - centroid, axis=1) for centroid in centroids]
        )
        left_preference = distances[:, 1] - distances[:, 0]
        left_rows = rows[np.argsort(left_preference)[-2:]]
        stage2_channel_labels[rows] = 1
        stage2_channel_labels[left_rows] = 0
        repaired_networks.append(int(network_id))

    label_by_name = {
        "dSPN_left": 0,
        "iSPN_left": 1,
        "dSPN_right": 2,
        "iSPN_right": 3,
    }
    stage2_labels = np.empty(len(feature_table), dtype=int)
    diagnostics: list[dict[str, Any]] = []

    for network_id in np.unique(network_ids):
        rows = np.flatnonzero(network_ids == network_id)
        left_rows = rows[stage2_channel_labels[rows] == 0]
        right_rows = rows[stage2_channel_labels[rows] == 1]
        ordered_rows = np.concatenate([left_rows, right_rows])

        unit_keys = feature_table.iloc[ordered_rows]["unit_key"].tolist()
        network_vectors = (
            vector_table.loc[vector_table["unit_key"].isin(unit_keys)]
            .pivot(index="unit_key", columns="sample_index", values="z_rate")
            .reindex(unit_keys)
        )
        # Networks may retain different numbers of trials after excluding
        # phase-0 trials shorter than six bins. Within one network, however,
        # all four SPN vectors share the same retained trials and length.
        network_vectors = network_vectors.loc[:, network_vectors.notna().all(axis=0)]
        vectors = network_vectors.to_numpy(float)
        corr = np.corrcoef(vectors)
        left_scores = corr[:2, 2:].mean(axis=1)
        right_scores = corr[2:, :2].mean(axis=1)

        d_left = _direct_candidate(left_scores, left_rows, 0, x12_z)
        d_right = _direct_candidate(right_scores, right_rows, 1, x12_z)
        i_left = 1 - d_left
        i_right = 1 - d_right

        stage2_labels[left_rows[d_left]] = label_by_name["dSPN_left"]
        stage2_labels[left_rows[i_left]] = label_by_name["iSPN_left"]
        stage2_labels[right_rows[d_right]] = label_by_name["dSPN_right"]
        stage2_labels[right_rows[i_right]] = label_by_name["iSPN_right"]

        diagnostics.append(
            {
                "network_id": int(network_id),
                "left_d_cross_score": float(left_scores[d_left]),
                "left_i_cross_score": float(left_scores[i_left]),
                "right_d_cross_score": float(right_scores[d_right]),
                "right_i_cross_score": float(right_scores[i_right]),
                "exact_dSPN_pair": bool(
                    true_names[left_rows[d_left]] == "dSPN_left"
                    and true_names[right_rows[d_right]] == "dSPN_right"
                ),
            }
        )

    true_stage2 = np.array([label_by_name[name] for name in true_names], dtype=int)
    return {
        "stage1_labels": stage1_labels,
        "stage2_channel_labels": stage2_channel_labels,
        "stage2_labels": stage2_labels,
        "stage1_ari": adjusted_rand_score(true_channel, stage1_labels),
        "stage2_ari": adjusted_rand_score(true_stage2, stage2_labels),
        "stage1_accuracy": float(np.mean(true_channel == stage1_labels)),
        "stage2_accuracy": float(np.mean(true_stage2 == stage2_labels)),
        "stage1_confusion": confusion_matrix(
            true_channel,
            stage1_labels,
            labels=[0, 1],
            normalize="true",
        ),
        "stage2_confusion": confusion_matrix(
            true_stage2,
            stage2_labels,
            labels=[0, 1, 2, 3],
            normalize="true",
        ),
        "repaired_networks": repaired_networks,
        "stage2_diagnostics": pd.DataFrame(diagnostics),
    }


def unit_profile_table(result: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for unit_i, unit_id in enumerate(result["unit_ids"]):
        for j in range(2 * LAST_K):
            rows.append(
                {
                    "session": result["session"],
                    "unit_id": int(unit_id),
                    "spn_name": str(result["final_names"][unit_i]),
                    "choice_side": "left" if j < LAST_K else "right",
                    "bin_from_decision": int(-(LAST_K - (j % LAST_K))),
                    "z_rate": float(result["x12_z"][unit_i, j]),
                }
            )
    return pd.DataFrame(rows)


def correlation_table(result: dict[str, Any]) -> pd.DataFrame:
    rows = []
    label_names = [result["name_map"][i] for i in range(4)]
    all_trials = np.arange(len(result["decision_time_ms"]))
    fast, slow = split_fast_slow(np.asarray(result["decision_time_ms"], float), all_trials)
    metadata = {
        "fast": (len(fast), float(np.max(np.asarray(result["decision_time_ms"])[fast]))),
        "slow": (len(slow), float(np.min(np.asarray(result["decision_time_ms"])[slow]))),
    }
    for speed, matrix in (("fast", result["corr_fast"]), ("slow", result["corr_slow"])):
        n_trials, cutoff_ms = metadata[speed]
        for i, name_i in enumerate(label_names):
            for j, name_j in enumerate(label_names):
                rows.append(
                    {
                        "session": result["session"],
                        "speed": speed,
                        "n_trials": n_trials,
                        "decision_time_cutoff_ms": cutoff_ms,
                        "spn_i": name_i,
                        "spn_j": name_j,
                        "pearson_r": float(matrix[i, j]),
                    }
                )
    return pd.DataFrame(rows)


def population_activity_table(result: dict[str, Any]) -> pd.DataFrame:
    rates = result["rates"]
    centers = np.asarray(result["bin_centers_s"], float)
    rows = []
    for trial_i, trial_id in enumerate(result["trial_ids"]):
        rt_s = float(result["decision_time_ms"][trial_i]) / 1000.0
        bins = np.flatnonzero((centers >= -rt_s) & (centers < -float(result.get("premove_gap_s", 0.010))))
        for order, bin_i in enumerate(bins):
            row = {
                "session": result["session"],
                "trial_id": int(trial_id),
                "bin_index": int(order),
                "time_to_decision_s": float(centers[bin_i]),
                "choice_left": int(result["choice_left"][trial_i]),
                "decision_time_ms": float(result["decision_time_ms"][trial_i]),
            }
            for name in SPN_NAMES:
                units = np.flatnonzero(result["final_names"] == name)
                row[name] = float(rates[units, trial_i, bin_i].mean())
            rows.append(row)
    return pd.DataFrame(rows)


_EXACT_EPS = 1e-8
_EXACT_SPN_DISPLAY_ORDER = (
    "dSPN_left",
    "dSPN_right",
    "iSPN_left",
    "iSPN_right",
)


def apply_fsi_audit_to_fixed_labels(
    labels: pd.DataFrame,
    audit: pd.DataFrame,
    match_columns: list[str],
    minimum_units_per_population: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach an FSI audit to fixed clustering labels without reclustering.

    Only rows explicitly marked ``strongly_fsi_like`` are removed from the
    downstream CLAW unit set. All other assigned units retain their original
    ``final_label`` and ``final_name`` values. The default of one is only a
    nonempty-population guard and does not apply the former three-unit session
    exclusion.
    """
    required_label_columns = {
        "session",
        "unit_id",
        "final_label",
        "final_name",
        *match_columns,
    }
    required_audit_columns = {
        "strongly_fsi_like",
        *match_columns,
    }
    missing_label_columns = required_label_columns.difference(labels.columns)
    missing_audit_columns = required_audit_columns.difference(audit.columns)
    if missing_label_columns:
        raise KeyError(f"Fixed-label table is missing: {sorted(missing_label_columns)}")
    if missing_audit_columns:
        raise KeyError(f"FSI audit is missing: {sorted(missing_audit_columns)}")

    fixed = labels.loc[
        (pd.to_numeric(labels["final_label"], errors="coerce") >= 0)
        & labels["final_name"].notna()
        & labels["final_name"].ne("unassigned")
    ].copy()
    audited = audit.copy()
    if fixed.duplicated(match_columns).any():
        raise ValueError(f"Fixed labels contain duplicate keys: {match_columns}")
    if audited.duplicated(match_columns).any():
        raise ValueError(f"FSI audit contains duplicate keys: {match_columns}")

    audit_columns = [*match_columns, "strongly_fsi_like"]
    if "final_name" in audited.columns:
        audit_columns.append("final_name")
    audited = audited[audit_columns].rename(
        columns={"final_name": "audited_final_name"}
    )
    merged = fixed.merge(
        audited,
        on=match_columns,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    missing_from_audit = merged["_merge"].eq("left_only")
    unexpected_in_audit = merged["_merge"].eq("right_only")
    if missing_from_audit.any() or unexpected_in_audit.any():
        raise ValueError(
            "The FSI audit does not exactly match the assigned fixed labels: "
            f"{int(missing_from_audit.sum())} missing and "
            f"{int(unexpected_in_audit.sum())} unexpected units."
        )
    merged = merged.drop(columns="_merge")

    if "audited_final_name" in merged.columns:
        label_mismatch = merged["audited_final_name"].astype(str).ne(
            merged["final_name"].astype(str)
        )
        if label_mismatch.any():
            raise ValueError(
                "The audit subtype names do not match the saved fixed labels."
            )
        merged = merged.drop(columns="audited_final_name")

    audit_values = merged["strongly_fsi_like"].astype(str).str.strip().str.lower()
    invalid_values = ~audit_values.isin(["true", "false"])
    if invalid_values.any():
        invalid = sorted(merged.loc[invalid_values, "strongly_fsi_like"].astype(str).unique())
        raise ValueError(f"Invalid strongly_fsi_like values: {invalid}")
    merged["strongly_fsi_like"] = audit_values.eq("true")
    merged["retained_after_fsi_audit"] = ~merged["strongly_fsi_like"]

    summaries: list[dict[str, Any]] = []
    session_inclusion: dict[str, bool] = {}
    for session, session_labels in merged.groupby("session", sort=False):
        original_counts = session_labels["final_name"].value_counts()
        retained = session_labels.loc[session_labels["retained_after_fsi_audit"]]
        retained_counts = retained["final_name"].value_counts()
        remaining = retained_counts.reindex(
            _EXACT_SPN_DISPLAY_ORDER, fill_value=0
        ).astype(int)
        included = bool(remaining.min() >= int(minimum_units_per_population))
        session_inclusion[str(session)] = included
        row: dict[str, Any] = {
            "session": str(session),
            "fsi_audit_applied": True,
            "fixed_labels_preserved": True,
            "minimum_units_per_population": int(minimum_units_per_population),
            "included_in_claw": included,
            "n_units_assigned": int(len(session_labels)),
            "n_units_excluded_as_fsi": int(session_labels["strongly_fsi_like"].sum()),
            "n_units_for_claw": int(len(retained)),
            "exclusion_reason": (
                "" if included else "fewer_than_minimum_units_in_at_least_one_population"
            ),
        }
        for population in _EXACT_SPN_DISPLAY_ORDER:
            row[f"n_{population}"] = int(original_counts.get(population, 0))
            row[f"n_{population}_for_claw"] = int(
                retained_counts.get(population, 0)
            )
        summaries.append(row)

    merged["session_included_in_claw"] = (
        merged["session"].astype(str).map(session_inclusion).astype(bool)
    )
    merged["used_in_claw"] = (
        merged["retained_after_fsi_audit"]
        & merged["session_included_in_claw"]
    )
    return merged, pd.DataFrame(summaries)


def _exact_causal_kernel(
    sigma_bins: float,
    truncate: float = 4.0,
    epsilon: float = _EXACT_EPS,
) -> np.ndarray:
    if sigma_bins <= 0:
        return np.array([1.0], dtype=float)
    time = np.arange(0, int(np.ceil(truncate * sigma_bins)) + 1)
    kernel = np.exp(-0.5 * (time / sigma_bins) ** 2)
    return (kernel / (kernel.sum() + epsilon)).astype(float)


def _exact_causal_smooth(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.convolve(values, kernel, mode="full")[: len(values)]


def _exact_corr_matrix(cluster_vectors: list[np.ndarray | None]) -> np.ndarray:
    matrix = np.full((4, 4), np.nan, dtype=float)
    valid = [label for label in range(4) if cluster_vectors[label] is not None]
    if not valid:
        return matrix
    if len(valid) == 1:
        matrix[valid[0], valid[0]] = 1.0
        return matrix
    lengths = [len(cluster_vectors[label]) for label in valid]
    if len(set(lengths)) != 1:
        return matrix
    submatrix = np.corrcoef(np.vstack([cluster_vectors[label] for label in valid]))
    for row_index, row_label in enumerate(valid):
        for col_index, col_label in enumerate(valid):
            matrix[row_label, col_label] = submatrix[row_index, col_index]
    for label in valid:
        matrix[label, label] = 1.0
    return matrix


def _label_name_map(final_labels: np.ndarray, final_names: np.ndarray) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for label in range(4):
        rows = np.flatnonzero(np.asarray(final_labels, dtype=int) == label)
        mapping[label] = (
            str(np.asarray(final_names, dtype=object)[rows[0]])
            if rows.size
            else f"cluster_{label}"
        )
    return mapping


def _stein_compute_unit_rate_exact(
    unit_spikes: np.ndarray,
    event_times: np.ndarray,
    trial_mask: np.ndarray,
    window: tuple[float, float],
) -> float:
    event_times = np.asarray(event_times).reshape(-1).astype(float)
    trial_mask = np.asarray(trial_mask).reshape(-1).astype(bool)
    start, stop = window
    duration = float(stop - start)
    if duration <= 0:
        raise ValueError("Window duration must be positive.")
    trials = np.where(trial_mask)[0]
    if trials.size == 0:
        return 0.0
    total = 0
    for trial in trials:
        left = event_times[trial] + start
        right = event_times[trial] + stop
        total += np.sum((unit_spikes >= left) & (unit_spikes < right))
    return total / (trials.size * duration + _EXACT_EPS)


def _stein_trial_rates_exact(
    unit_spikes: np.ndarray,
    event_times: np.ndarray,
    trial_indices: np.ndarray,
    window: tuple[float, float],
) -> np.ndarray:
    event_times = np.asarray(event_times).reshape(-1).astype(float)
    trial_indices = np.asarray(trial_indices, dtype=int).reshape(-1)
    start, stop = window
    duration = float(stop - start)
    if duration <= 0:
        raise ValueError("Window duration must be positive.")
    rates = np.full(trial_indices.size, np.nan, dtype=float)
    for index, trial in enumerate(trial_indices):
        left = event_times[trial] + start
        right = event_times[trial] + stop
        rates[index] = np.sum((unit_spikes >= left) & (unit_spikes < right)) / (
            duration + _EXACT_EPS
        )
    return rates


def _stein_safe_wilcoxon_exact(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import wilcoxon

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 5 or np.allclose(x - y, 0.0):
        return 1.0
    try:
        return float(
            wilcoxon(
                x,
                y,
                alternative="two-sided",
                zero_method="wilcox",
            ).pvalue
        )
    except Exception:
        return 1.0


def _stein_safe_ranksums_exact(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import ranksums

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size < 5 or y.size < 5:
        return 1.0
    try:
        return float(ranksums(x, y).pvalue)
    except Exception:
        return 1.0


def _stein_passes_first_modulation_filter_exact(
    unit_spikes: np.ndarray,
    stimulus_times: np.ndarray,
    movement_onsets: np.ndarray,
    baseline_trials: np.ndarray,
    trials_left: np.ndarray,
    trials_right: np.ndarray,
) -> bool:
    all_trials = np.where(
        np.asarray(baseline_trials, dtype=bool)
        & np.isfinite(np.asarray(stimulus_times, dtype=float))
    )[0]
    if all_trials.size >= 10:
        baseline = _stein_trial_rates_exact(
            unit_spikes, stimulus_times, all_trials, (-0.20, 0.00)
        )
        trial_rate = _stein_trial_rates_exact(
            unit_spikes, stimulus_times, all_trials, (0.00, 0.40)
        )
        stimulus_rate = _stein_trial_rates_exact(
            unit_spikes, stimulus_times, all_trials, (0.05, 0.15)
        )
        p_trial = _stein_safe_wilcoxon_exact(trial_rate, baseline)
        p_stimulus = _stein_safe_wilcoxon_exact(stimulus_rate, baseline)
    else:
        p_trial = 1.0
        p_stimulus = 1.0

    if len(trials_left) + len(trials_right) >= 10:
        movement_trials = np.sort(
            np.concatenate([trials_left, trials_right])
        ).astype(int)
        movement_trials = movement_trials[
            np.isfinite(np.asarray(movement_onsets)[movement_trials])
        ]
        if movement_trials.size >= 10:
            baseline_movement = _stein_trial_rates_exact(
                unit_spikes, stimulus_times, movement_trials, (-0.20, 0.00)
            )
            premovement = _stein_trial_rates_exact(
                unit_spikes, movement_onsets, movement_trials, (-0.10, 0.05)
            )
            p_movement = _stein_safe_wilcoxon_exact(
                premovement, baseline_movement
            )
        else:
            p_movement = 1.0

        left = np.asarray(trials_left, dtype=int)[
            np.isfinite(np.asarray(movement_onsets)[np.asarray(trials_left, dtype=int)])
        ]
        right = np.asarray(trials_right, dtype=int)[
            np.isfinite(np.asarray(movement_onsets)[np.asarray(trials_right, dtype=int)])
        ]
        premovement_left = (
            _stein_trial_rates_exact(
                unit_spikes, movement_onsets, left, (-0.10, 0.05)
            )
            if left.size
            else np.array([])
        )
        premovement_right = (
            _stein_trial_rates_exact(
                unit_spikes, movement_onsets, right, (-0.10, 0.05)
            )
            if right.size
            else np.array([])
        )
        p_choice = _stein_safe_ranksums_exact(
            premovement_left, premovement_right
        )
    else:
        p_movement = 1.0
        p_choice = 1.0

    return (
        (p_trial < 0.05)
        or (p_stimulus < 0.05)
        or (p_movement < 0.05)
        or (p_choice < 0.05)
    )


def _stein_counts_in_windows_exact(
    sorted_spikes: np.ndarray,
    starts: np.ndarray,
    stops: np.ndarray,
) -> np.ndarray:
    left = np.searchsorted(sorted_spikes, starts, side="left")
    right = np.searchsorted(sorted_spikes, stops, side="left")
    return (right - left).astype(int)


def _stein_firing_rate_by_trial_exact(
    sorted_spikes: np.ndarray,
    event_times: np.ndarray,
    trial_indices: np.ndarray,
    window: tuple[float, float],
) -> np.ndarray:
    event_times = np.asarray(event_times).reshape(-1).astype(float)
    trial_indices = np.asarray(trial_indices).reshape(-1).astype(int)
    start, stop = window
    duration = float(stop - start)
    if duration <= 0:
        raise ValueError("Window duration must be positive.")
    starts = event_times[trial_indices] + start
    stops = event_times[trial_indices] + stop
    valid = np.isfinite(starts) & np.isfinite(stops)
    starts = starts[valid]
    stops = stops[valid]
    if starts.size == 0:
        return np.zeros(0, dtype=float)
    return _stein_counts_in_windows_exact(sorted_spikes, starts, stops).astype(
        float
    ) / (duration + _EXACT_EPS)


def _stein_is_task_modulated_exact(
    unit_spikes: np.ndarray,
    stimulus_times: np.ndarray,
    movement_onsets: np.ndarray,
    choice: np.ndarray,
    included: np.ndarray,
    usable: np.ndarray,
    choice_left_code: int,
    choice_right_code: int,
) -> bool:
    from scipy.stats import ranksums, wilcoxon

    stimulus_times = np.asarray(stimulus_times).squeeze().reshape(-1)
    movement_onsets = np.asarray(movement_onsets).squeeze().reshape(-1)
    choice = np.asarray(choice).squeeze().reshape(-1)
    included = np.asarray(included, dtype=bool).squeeze().reshape(-1)
    usable = np.asarray(usable, dtype=bool).squeeze().reshape(-1)

    n_trials = min(
        len(stimulus_times), len(movement_onsets), len(choice), len(included)
    )
    stimulus_times = stimulus_times[:n_trials]
    movement_onsets = movement_onsets[:n_trials]
    choice = choice[:n_trials]
    included = included[:n_trials]
    if len(usable) != n_trials:
        usable = included & np.isfinite(stimulus_times) & np.isfinite(
            movement_onsets
        )
    else:
        usable = usable[:n_trials]

    included_trials = np.where(included & np.isfinite(stimulus_times))[0]
    if included_trials.size < 20:
        return True

    baseline = _stein_firing_rate_by_trial_exact(
        unit_spikes, stimulus_times, included_trials, (-0.20, 0.00)
    )
    trial_rate = _stein_firing_rate_by_trial_exact(
        unit_spikes, stimulus_times, included_trials, (0.00, 0.40)
    )
    stimulus_rate = _stein_firing_rate_by_trial_exact(
        unit_spikes, stimulus_times, included_trials, (0.05, 0.15)
    )

    keep = False
    try:
        if baseline.size >= 10 and trial_rate.size == baseline.size:
            keep = keep or (wilcoxon(baseline, trial_rate).pvalue < 0.05)
    except Exception:
        pass
    try:
        if baseline.size >= 10 and stimulus_rate.size == baseline.size:
            keep = keep or (wilcoxon(baseline, stimulus_rate).pvalue < 0.05)
    except Exception:
        pass

    movement_trials = np.where(usable & np.isfinite(movement_onsets))[0]
    if movement_trials.size >= 10:
        baseline_movement = _stein_firing_rate_by_trial_exact(
            unit_spikes, stimulus_times, movement_trials, (-0.20, 0.00)
        )
        premovement = _stein_firing_rate_by_trial_exact(
            unit_spikes, movement_onsets, movement_trials, (-0.10, 0.05)
        )
        try:
            keep |= wilcoxon(baseline_movement, premovement).pvalue < 0.05
        except Exception:
            pass

        left = movement_trials[choice[movement_trials] == choice_left_code]
        right = movement_trials[choice[movement_trials] == choice_right_code]
        if left.size >= 5 and right.size >= 5:
            premovement_left = _stein_firing_rate_by_trial_exact(
                unit_spikes, movement_onsets, left, (-0.10, 0.05)
            )
            premovement_right = _stein_firing_rate_by_trial_exact(
                unit_spikes, movement_onsets, right, (-0.10, 0.05)
            )
            try:
                keep |= ranksums(premovement_left, premovement_right).pvalue < 0.05
            except Exception:
                pass
    return bool(keep)


def _stein_geometry_exact(premove_gap_s: float) -> dict[str, np.ndarray]:
    edges = np.arange(-0.500, 0.200 + 1e-12, 0.010)
    centers = (edges[:-1] + edges[1:]) / 2
    right_edges = edges[1:]
    valid_pre = np.where(right_edges <= (-premove_gap_s + 1e-12))[0]
    stage_a = valid_pre[-6:]
    stage_b = np.where((centers >= -0.10) & (centers < 0.05))[0]
    kernel = _exact_causal_kernel(0.025 / 0.010, epsilon=_EXACT_EPS)
    return {
        "edges": edges,
        "centers": centers,
        "stage_a": stage_a,
        "stage_b": stage_b,
        "kernel": kernel,
    }


def _stein_extract_feature_exact(
    unit_spikes: np.ndarray,
    movement_time: float,
    normalization: float,
    geometry: dict[str, np.ndarray],
    bin_indices: np.ndarray,
) -> np.ndarray:
    relative = unit_spikes - movement_time
    relative = relative[(relative >= -0.500) & (relative < 0.200)]
    counts, _ = np.histogram(relative, bins=geometry["edges"])
    firing_rate = counts / 0.010
    smoothed = _exact_causal_smooth(firing_rate, geometry["kernel"])
    return (smoothed[np.asarray(bin_indices, dtype=int)] / (
        normalization + _EXACT_EPS
    )).astype(float)


def _stein_recluster_exact(
    distance: np.ndarray,
    minimum_size: int = 3,
    max_iterations: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    kept = np.arange(distance.shape[0])
    for _ in range(max_iterations):
        if kept.size < 2 * minimum_size:
            break
        subdistance = distance[np.ix_(kept, kept)]
        labels = AgglomerativeClustering(
            n_clusters=2,
            linkage="average",
            metric="precomputed",
        ).fit_predict(subdistance)
        counts = np.bincount(labels, minlength=2)
        if counts.min() >= minimum_size:
            return kept, labels
        kept = kept[labels != int(np.argmin(counts))]
    if kept.size >= 2 * minimum_size:
        subdistance = distance[np.ix_(kept, kept)]
        labels = AgglomerativeClustering(
            n_clusters=2,
            linkage="average",
            metric="precomputed",
        ).fit_predict(subdistance)
        return kept, labels
    return kept, np.zeros(kept.size, dtype=int)


def _stein_split_fast_slow_exact(
    trials: np.ndarray,
    reaction_time_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    trials = np.asarray(trials, dtype=int).reshape(-1)
    reaction_time_s = np.asarray(reaction_time_s, dtype=float).reshape(-1)
    values = reaction_time_s[trials]
    valid = np.isfinite(values)
    trials = trials[valid]
    values = values[valid]
    if trials.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int), {}
    order = np.lexsort((trials, values))
    trials = trials[order]
    values = values[order]
    cut = len(trials) // 2
    fast = trials[:cut]
    slow = trials[cut:]
    metadata = {
        "fast_min_s": float(values[0]) if cut else np.nan,
        "fast_max_s": float(values[cut - 1]) if cut else np.nan,
        "slow_min_s": float(values[cut]) if cut < len(values) else np.nan,
    }
    return fast, slow, metadata


def _stein_cluster_vectors_exact(
    result: dict[str, Any],
    trial_list: np.ndarray,
) -> list[np.ndarray | None]:
    payload = result["source_payload"]
    spike_times = np.asarray(payload["spike_times"], dtype=float)
    spike_clusters = np.asarray(payload["spike_clusters"], dtype=int)
    stimulus_times = np.asarray(payload["stimulus_times"], dtype=float)
    movement_onsets = np.asarray(payload["movement_onsets"], dtype=float)
    baseline_trials = np.asarray(payload["baseline_trials"], dtype=bool)
    unit_ids = np.asarray(result["unit_ids"], dtype=int)
    final_labels = np.asarray(result["final_labels"], dtype=int)
    geometry = result["geometry"]

    spikes_by_unit: dict[int, np.ndarray] = {}
    normalization_by_unit: dict[int, float] = {}
    for unit_id in unit_ids:
        unit_spikes = np.sort(spike_times[spike_clusters == unit_id])
        spikes_by_unit[int(unit_id)] = unit_spikes
        baseline_rate = _stein_compute_unit_rate_exact(
            unit_spikes,
            stimulus_times,
            baseline_trials,
            (-0.20, 0.00),
        )
        normalization_by_unit[int(unit_id)] = baseline_rate + 1.0

    vectors: list[np.ndarray | None] = []
    for label in range(4):
        unit_rows = np.where(final_labels == label)[0]
        if unit_rows.size == 0:
            vectors.append(None)
            continue
        trial_means: list[np.ndarray] = []
        for trial in np.asarray(trial_list, dtype=int):
            movement_time = movement_onsets[trial]
            if not np.isfinite(movement_time):
                continue
            segments = []
            for row in unit_rows:
                unit_id = int(unit_ids[row])
                segment = _stein_extract_feature_exact(
                    spikes_by_unit[unit_id],
                    movement_time,
                    normalization_by_unit[unit_id],
                    geometry,
                    geometry["stage_a"],
                )
                if np.all(np.isfinite(segment)):
                    segments.append(segment)
            if segments:
                trial_means.append(np.mean(np.vstack(segments), axis=0))
        if not trial_means:
            vectors.append(None)
            continue
        vector = np.concatenate(trial_means)
        if not np.all(np.isfinite(vector)) or np.std(vector) < 1e-6:
            vectors.append(None)
            continue
        vectors.append(
            (vector - vector.mean()) / (vector.std() + _EXACT_EPS)
        )
    return vectors




def steinmetz_unit_profile_table_exact(result: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for unit_row, unit_id in enumerate(result["unit_ids"]):
        if int(result["final_labels"][unit_row]) < 0:
            continue
        for feature_index, value in enumerate(result["x12_z"][unit_row]):
            rows.append(
                {
                    "session": result["session"],
                    "unit_id": int(unit_id),
                    "spn_name": str(result["final_names"][unit_row]),
                    "choice_side": "left" if feature_index < 6 else "right",
                    "bin_from_decision": int(-6 + (feature_index % 6)),
                    "z_rate": float(value),
                }
            )
    return pd.DataFrame(rows)


def _exact_correlation_table(result: dict[str, Any]) -> pd.DataFrame:
    name_map = _label_name_map(result["final_labels"], result["final_names"])
    rows: list[dict[str, Any]] = []
    split_metadata = result["split_metadata"]
    metadata = {
        "fast": (
            len(result["fast_trials"]),
            1000.0 * float(split_metadata.get("fast_max_s", np.nan)),
        ),
        "slow": (
            len(result["slow_trials"]),
            1000.0 * float(split_metadata.get("slow_min_s", np.nan)),
        ),
    }
    for speed, matrix in (
        ("fast", result["corr_fast"]),
        ("slow", result["corr_slow"]),
    ):
        n_trials, cutoff = metadata[speed]
        for row_label in range(4):
            for col_label in range(4):
                rows.append(
                    {
                        "session": result["session"],
                        "speed": speed,
                        "n_trials": int(n_trials),
                        "decision_time_cutoff_ms": float(cutoff),
                        "spn_i": name_map[row_label],
                        "spn_j": name_map[col_label],
                        "pearson_r": float(matrix[row_label, col_label]),
                    }
                )
    return pd.DataFrame(rows)


def steinmetz_correlation_table_exact(result: dict[str, Any]) -> pd.DataFrame:
    return _exact_correlation_table(result)


def steinmetz_population_activity_table_exact(
    result: dict[str, Any],
    claw_seed: int = 2000,
    minimum_units_per_population: int = 1,
    excluded_unit_ids: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Reproduce the activity table, optionally excluding post-clustering units."""
    from .datasets import _steinmetz_match_trials_exact

    payload = result["source_payload"]
    session_dir = Path(payload["session_dir"])
    final_names = np.asarray(result["final_names"], dtype=object)
    unit_ids = np.asarray(result["unit_ids"], dtype=int)
    if excluded_unit_ids is not None:
        excluded_unit_ids = np.asarray(list(excluded_unit_ids), dtype=int)
        keep = ~np.isin(unit_ids, excluded_unit_ids)
        final_names = final_names[keep]
        unit_ids = unit_ids[keep]
    populations = _EXACT_SPN_DISPLAY_ORDER
    population_units = {
        population: unit_ids[final_names == population]
        for population in populations
    }
    if min(len(units) for units in population_units.values()) < int(
        minimum_units_per_population
    ):
        population_counts = {
            population: len(units) for population, units in population_units.items()
        }
        raise ValueError(
            f"{result['session']}: fewer than {minimum_units_per_population} units "
            "in at least one SPN population for the CLAW analysis. "
            f"Remaining counts: {population_counts}."
        )

    stimulus_times = np.asarray(payload["stimulus_times"], dtype=float)
    movement_onsets = np.asarray(payload["movement_onsets"], dtype=float)
    reaction_time = np.asarray(payload["reaction_time_s"], dtype=float)
    choice = np.asarray(payload["choice"], dtype=int)
    included = np.asarray(payload["included"], dtype=bool)
    movement_intervals = np.asarray(payload["movement_intervals"], dtype=float)
    early_movement = np.zeros(len(stimulus_times), dtype=bool)
    move_starts = movement_intervals[:, 0]
    move_ends = movement_intervals[:, 1]
    for trial, (start, stop) in enumerate(
        zip(stimulus_times - 0.05, stimulus_times + 0.125)
    ):
        if not (np.isfinite(start) and np.isfinite(stop)):
            early_movement[trial] = True
        else:
            early_movement[trial] = np.any(
                (move_starts < stop) & (move_ends > start)
            )
    usable = (
        included
        & np.isfinite(reaction_time)
        & (reaction_time >= 0.125)
        & (reaction_time <= 0.400)
        & (~early_movement)
        & (choice != 0)
    )
    trials_left = np.where(
        usable & (choice == int(payload["choice_left_code"]))
    )[0]
    trials_right = np.where(
        usable & (choice == int(payload["choice_right_code"]))
    )[0]
    trials_left, trials_right, _ = _steinmetz_match_trials_exact(
        session_dir,
        trials_left,
        trials_right,
        included,
        matching_seed=int(claw_seed),
        n_bins=3,
        minimum_per_bin=5,
    )
    trials_all = np.sort(np.concatenate([trials_left, trials_right]))

    spike_times = np.load(session_dir / "spikes.times.npy").astype(float)
    spike_clusters = np.load(session_dir / "spikes.clusters.npy").astype(int)
    population_spikes = {
        population: np.sort(
            spike_times[np.isin(spike_clusters, population_units[population])]
        )
        for population in populations
    }
    kernel = _exact_causal_kernel(
        0.025 / (0.010 + 1e-12), epsilon=1e-12
    )

    rows: list[dict[str, Any]] = []
    for trial in trials_all:
        stimulus_time = float(stimulus_times[trial])
        movement_time = float(movement_onsets[trial])
        rt = float(reaction_time[trial])
        n_bins = int(np.floor((rt - 0.010 + 1e-12) / 0.010))
        if n_bins <= 0 or not np.isfinite(rt):
            continue
        time_from_stimulus = np.arange(n_bins, dtype=float) * 0.010
        edges = np.arange(n_bins + 1, dtype=float) * 0.010
        trial_rates: dict[str, np.ndarray] = {}
        for population in populations:
            spikes = population_spikes[population]
            stop = stimulus_time + n_bins * 0.010
            selected = spikes[(spikes >= stimulus_time) & (spikes <= stop)]
            counts, _ = np.histogram(selected - stimulus_time, bins=edges)
            rate = counts.astype(float) / (0.010 + 1e-12)
            rate = _exact_causal_smooth(rate, kernel)
            trial_rates[population] = rate / len(population_units[population])

        for bin_index in range(n_bins):
            row = {
                "session": result["session"],
                "trial_id": int(trial),
                "bin_index": int(bin_index),
                "time_from_stimulus_s": float(time_from_stimulus[bin_index]),
                "time_to_decision_s": float(
                    time_from_stimulus[bin_index] - rt
                ),
                "choice_left": int(
                    choice[trial] == int(payload["choice_left_code"])
                ),
                "decision_time_ms": float(1000.0 * rt),
            }
            for population in populations:
                row[population] = float(trial_rates[population][bin_index])
            rows.append(row)
    return pd.DataFrame(rows)


def _ibl_stage_indices_exact(
    bin_edges_s: np.ndarray,
    premove_gap_s: float = 0.010,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.asarray(bin_edges_s, dtype=float)
    centers = (edges[:-1] + edges[1:]) / 2
    right_edges = edges[1:]
    valid_pre = np.flatnonzero(
        right_edges <= (-premove_gap_s + _EXACT_EPS)
    )
    if valid_pre.size < 6:
        raise ValueError("Not enough premovement bins for the IBL 12D feature.")
    stage_a = valid_pre[-6:]
    stage_b = (centers >= -0.10) & (centers < 0.05)
    if not np.any(stage_b):
        raise ValueError("IBL Stage B window is outside the firing-rate axis.")
    return centers, stage_a, stage_b


def _ibl_recluster_exact(
    distance: np.ndarray,
    minimum_size: int = 3,
    max_iterations: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    kept = np.arange(distance.shape[0])
    for _ in range(max_iterations):
        if kept.size < 2 * minimum_size:
            break
        subdistance = distance[np.ix_(kept, kept)]
        labels = AgglomerativeClustering(
            n_clusters=2,
            linkage="average",
            metric="precomputed",
        ).fit_predict(subdistance)
        counts = np.bincount(labels, minlength=2)
        if counts.min() >= minimum_size:
            return kept, labels
        kept = kept[labels != int(np.argmin(counts))]
    if kept.size >= 2 * minimum_size:
        subdistance = distance[np.ix_(kept, kept)]
        labels = AgglomerativeClustering(
            n_clusters=2,
            linkage="average",
            metric="precomputed",
        ).fit_predict(subdistance)
        return kept, labels
    return kept, np.zeros(kept.size, dtype=int)


def _ibl_split_fast_slow_exact(
    trials: np.ndarray,
    reaction_time_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    trials = np.asarray(trials, dtype=int)
    reaction_time_s = np.asarray(reaction_time_s, dtype=float)
    values = reaction_time_s[trials]
    valid = np.isfinite(values)
    trials = trials[valid]
    values = values[valid]
    if trials.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int), {}
    order = np.argsort(values)
    trials = trials[order]
    values = values[order]
    cut = len(trials) // 2
    return (
        trials[:cut],
        trials[cut:],
        {
            "fast_min_s": float(values[0]) if cut else np.nan,
            "fast_max_s": float(values[cut - 1]) if cut else np.nan,
            "slow_min_s": float(values[cut]) if cut < len(values) else np.nan,
        },
    )


def _ibl_cluster_vectors_exact(
    result: dict[str, Any],
    trial_list: np.ndarray,
) -> list[np.ndarray | None]:
    trial_list = np.asarray(trial_list, dtype=int).reshape(-1)
    if trial_list.size == 0:
        return [None, None, None, None]
    rates = np.asarray(result["rates"], dtype=float)
    labels = np.asarray(result["final_labels"], dtype=int)
    stage_a = np.asarray(result["stage_a_bins"], dtype=int)
    vectors: list[np.ndarray | None] = []
    for label in range(4):
        units = np.where(labels == label)[0]
        if units.size == 0:
            vectors.append(None)
            continue
        trial_means = []
        for trial in trial_list:
            segment = rates[units, trial, :][:, stage_a]
            if segment.size:
                trial_means.append(segment.mean(axis=0))
        if not trial_means:
            vectors.append(None)
            continue
        vector = np.concatenate(trial_means)
        if not np.all(np.isfinite(vector)) or np.std(vector) < 1e-6:
            vectors.append(None)
            continue
        vectors.append(
            (vector - vector.mean()) / (vector.std() + _EXACT_EPS)
        )
    return vectors


def cluster_ibl_session_exact(payload: dict[str, Any]) -> dict[str, Any]:
    """Execute the attached IBL clustering notebook without simplification."""
    rates = np.asarray(payload["rates"], dtype=float)
    unit_ids = np.asarray(payload["unit_ids"], dtype=int)
    idx_left = np.asarray(payload["idxL"], dtype=int)
    idx_right = np.asarray(payload["idxR"], dtype=int)
    centers, stage_a, stage_b_mask = _ibl_stage_indices_exact(
        payload["bin_edges_s"], float(payload["premove_gap_s"])
    )

    left_mean = rates[:, idx_left, :][:, :, stage_a].mean(axis=1)
    right_mean = rates[:, idx_right, :][:, :, stage_a].mean(axis=1)
    stage_a_times = centers[stage_a]
    keep = np.ones(len(unit_ids), dtype=bool)
    for row in range(len(unit_ids)):
        slope_left = np.polyfit(stage_a_times, left_mean[row], 1)[0]
        slope_right = np.polyfit(stage_a_times, right_mean[row], 1)[0]
        if (
            not np.isfinite(slope_left)
            or not np.isfinite(slope_right)
            or slope_left < 1e-4
            or slope_right < 1e-4
        ):
            keep[row] = False

    unit_ids = unit_ids[keep]
    rates = rates[keep]
    left_mean = left_mean[keep]
    right_mean = right_mean[keep]
    if unit_ids.size < 8:
        raise ValueError(
            f"{payload['session']}: only {unit_ids.size} units after the original ramp filter."
        )

    x12 = np.concatenate([left_mean, right_mean], axis=1)
    x12_z = (x12 - x12.mean(axis=1, keepdims=True)) / (
        x12.std(axis=1, keepdims=True) + _EXACT_EPS
    )
    raw_channel_labels = AgglomerativeClustering(
        n_clusters=2, linkage="ward", metric="euclidean"
    ).fit_predict(x12_z)
    bias = x12[:, :6].mean(axis=1) - x12[:, 6:].mean(axis=1)
    bias_zero = (
        bias[raw_channel_labels == 0].mean()
        if np.any(raw_channel_labels == 0)
        else -np.inf
    )
    bias_one = (
        bias[raw_channel_labels == 1].mean()
        if np.any(raw_channel_labels == 1)
        else -np.inf
    )
    left_channel_label = 0 if bias_zero > bias_one else 1
    left_rows = np.where(raw_channel_labels == left_channel_label)[0]
    right_rows = np.where(raw_channel_labels != left_channel_label)[0]
    if left_rows.size < 2 or right_rows.size < 2:
        raise ValueError(
            f"{payload['session']}: Stage A produced a group smaller than two units."
        )

    all_trials = np.sort(np.concatenate([idx_left, idx_right]))
    if all_trials.size < 20:
        raise ValueError(f"{payload['session']}: fewer than 20 Stage B trials.")
    precomputed_vectors: list[np.ndarray | None] = [None] * len(unit_ids)
    for row in range(len(unit_ids)):
        vector = rates[row, all_trials, :][:, stage_b_mask].reshape(-1)
        if np.all(np.isfinite(vector)) and np.std(vector) >= 1e-6:
            precomputed_vectors[row] = (
                vector - vector.mean()
            ) / (vector.std() + _EXACT_EPS)

    def split_channel(channel_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        vectors: list[np.ndarray] = []
        kept: list[int] = []
        for row in channel_rows:
            vector = precomputed_vectors[row]
            if vector is None:
                continue
            vectors.append(vector)
            kept.append(int(row))
        kept_array = np.asarray(kept, dtype=int)
        if kept_array.size < 6:
            return kept_array, np.zeros(kept_array.size, dtype=int)

        correlation = np.corrcoef(np.vstack(vectors))
        correlation = np.nan_to_num(
            np.clip(correlation, -1.0, 1.0), nan=0.0
        )
        np.fill_diagonal(correlation, 1.0)
        distance = 1.0 - correlation
        np.fill_diagonal(distance, 0.0)
        kept_after_pruning, sublabels = _ibl_recluster_exact(
            distance, minimum_size=3
        )
        return kept_array[kept_after_pruning], sublabels

    kept_left, sublabels_left = split_channel(left_rows)
    kept_right, sublabels_right = split_channel(right_rows)
    final_labels = np.full(len(unit_ids), -1, dtype=int)
    final_names = np.array(["unassigned"] * len(unit_ids), dtype=object)
    for row, label in zip(kept_left, sublabels_left):
        final_labels[row] = int(label)
        final_names[row] = f"L_sub{label}"
    for row, label in zip(kept_right, sublabels_right):
        final_labels[row] = int(label) + 2
        final_names[row] = f"R_sub{label}"
    left_row_set = set(left_rows.tolist())
    lr_pref = np.array(
        ["L" if row in left_row_set else "R" for row in range(len(unit_ids))],
        dtype=object,
    )

    result = {
        "dataset": "IBL",
        "session": payload["session"],
        "session_path": payload["session_path"],
        "eid": payload["eid"],
        "pid": payload["pid"],
        "probe": payload["probe"],
        "unit_ids": unit_ids,
        "rates": rates,
        "trial_ids": np.asarray(payload["trial_ids"], dtype=int),
        "choice": np.asarray(payload["choice"], dtype=int),
        "choice_left": np.asarray(payload["choice_left"], dtype=bool),
        "decision_time_ms": np.asarray(payload["decision_time_ms"], dtype=float),
        "reaction_time_s": np.asarray(payload["reaction_time_s"], dtype=float),
        "signed_contrast": np.asarray(payload["signed_contrast"], dtype=float),
        "idxL": idx_left,
        "idxR": idx_right,
        "bin_edges_s": np.asarray(payload["bin_edges_s"], dtype=float),
        "bin_centers_s": centers,
        "stage_a_bins": stage_a,
        "stage_b_mask": stage_b_mask,
        "x12": x12,
        "x12_z": x12_z,
        "lr_pref": lr_pref,
        "raw_channel_labels": raw_channel_labels,
        "left_channel_label": int(left_channel_label),
        "final_labels": final_labels,
        "final_names": final_names,
        "source_payload": payload,
    }

    fast_trials, slow_trials, split_metadata = _ibl_split_fast_slow_exact(
        all_trials, result["reaction_time_s"]
    )
    corr_fast = _exact_corr_matrix(
        _ibl_cluster_vectors_exact(result, fast_trials)
    )
    corr_slow = _exact_corr_matrix(
        _ibl_cluster_vectors_exact(result, slow_trials)
    )
    delta = corr_slow - corr_fast
    best_pair: tuple[int, int] | None = None
    best_drop = np.inf
    for left_label in (0, 1):
        for right_label in (2, 3):
            drop = delta[left_label, right_label]
            if np.isfinite(drop) and drop < best_drop:
                best_drop = float(drop)
                best_pair = (left_label, right_label)

    name_map: dict[int, str] = {}
    if best_pair is not None:
        d_left, d_right = best_pair
        name_map = {
            int(d_left): "dSPN_left",
            int(1 - d_left): "iSPN_left",
            int(d_right): "dSPN_right",
            int(5 - d_right): "iSPN_right",
        }
        for label, name in name_map.items():
            final_names[final_labels == label] = name

    result.update(
        final_names=final_names,
        name_map=name_map,
        corr_fast=corr_fast,
        corr_slow=corr_slow,
        corr_drop=float(best_drop) if best_pair is not None else np.nan,
        fast_trials=fast_trials,
        slow_trials=slow_trials,
        split_metadata=split_metadata,
    )
    return result


def ibl_unit_profile_table_exact(result: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for unit_row, unit_id in enumerate(result["unit_ids"]):
        if int(result["final_labels"][unit_row]) < 0:
            continue
        for feature_index, value in enumerate(result["x12_z"][unit_row]):
            rows.append(
                {
                    "session": result["session"],
                    "unit_id": int(unit_id),
                    "spn_name": str(result["final_names"][unit_row]),
                    "choice_side": "left" if feature_index < 6 else "right",
                    "bin_from_decision": int(-6 + (feature_index % 6)),
                    "z_rate": float(value),
                }
            )
    return pd.DataFrame(rows)


def ibl_correlation_table_exact(result: dict[str, Any]) -> pd.DataFrame:
    return _exact_correlation_table(result)


def ibl_population_activity_table_exact(
    result: dict[str, Any],
    one: Any,
    claw_seed: int = 2000,
    minimum_units_per_population: int = 1,
    excluded_unit_ids: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Reproduce the firing-rate table, optionally excluding post-clustering units."""
    from brainbox.io.one import SpikeSortingLoader
    from brainbox.population.decode import get_spike_counts_in_bins
    from .datasets import (
        _exact_session_rng,
        _ibl_match_trials_by_signed_contrast_exact,
        _ibl_time_edges_exact,
        _ibl_trial_mask_exact,
        _make_intervals,
        _reorder_counts,
    )

    final_names = np.asarray(result["final_names"], dtype=object)
    unit_ids_all = np.asarray(result["unit_ids"], dtype=int)
    assigned = np.asarray(result["final_labels"], dtype=int) >= 0
    unit_ids = unit_ids_all[assigned]
    final_names = final_names[assigned]
    if excluded_unit_ids is not None:
        excluded_unit_ids = np.asarray(list(excluded_unit_ids), dtype=int)
        keep = ~np.isin(unit_ids, excluded_unit_ids)
        unit_ids = unit_ids[keep]
        final_names = final_names[keep]
    populations = _EXACT_SPN_DISPLAY_ORDER
    population_rows = {
        population: np.where(final_names == population)[0]
        for population in populations
    }
    if min(len(rows) for rows in population_rows.values()) < int(
        minimum_units_per_population
    ):
        population_counts = {
            population: len(rows) for population, rows in population_rows.items()
        }
        raise ValueError(
            f"{result['session']}: fewer than {minimum_units_per_population} units "
            "in at least one SPN population for the CLAW analysis. "
            f"Remaining counts: {population_counts}."
        )

    eid = result["eid"]
    pid = result["pid"]
    probe = result["probe"]
    trials = one.load_object(eid, "trials", collection="alf")
    mask = _ibl_trial_mask_exact(trials, 0.080, 0.400)
    choice_all = np.asarray(trials["choice"]).reshape(-1).astype(int)
    mask &= np.isin(choice_all, [-1, 1])
    trial_ids = np.where(mask)[0]
    stimulus = np.asarray(trials["stimOn_times"], dtype=float)[trial_ids]
    first_movement = np.asarray(
        trials["firstMovement_times"], dtype=float
    )[trial_ids]
    reaction_time = first_movement - stimulus
    choice = choice_all[trial_ids]
    contrast_left = np.nan_to_num(
        np.asarray(
            trials.get("contrastLeft", np.zeros_like(choice_all, dtype=float))
        )[trial_ids].astype(float),
        nan=0.0,
    )
    contrast_right = np.nan_to_num(
        np.asarray(
            trials.get("contrastRight", np.zeros_like(choice_all, dtype=float))
        )[trial_ids].astype(float),
        nan=0.0,
    )
    signed_contrast = contrast_right - contrast_left

    loader = SpikeSortingLoader(pid=pid, one=one)
    loader.load_channels()
    spikes, clusters, channels = loader.load_spike_sorting()
    clusters = loader.merge_clusters(spikes, clusters, channels)
    spike_times = np.asarray(spikes["times"], dtype=float)
    spike_clusters = np.asarray(spikes["clusters"], dtype=int)
    keep_spikes = np.isin(spike_clusters, unit_ids)
    spike_times = spike_times[keep_spikes]
    spike_clusters = spike_clusters[keep_spikes]

    baseline_intervals = np.column_stack([stimulus - 0.20, stimulus])
    baseline_counts, baseline_ids = get_spike_counts_in_bins(
        spike_times, spike_clusters, baseline_intervals
    )
    baseline_counts = _reorder_counts(
        baseline_counts, baseline_ids, unit_ids
    )
    baseline_rate = baseline_counts.mean(axis=1) / 0.20

    edges = _ibl_time_edges_exact()
    intervals = _make_intervals(first_movement, edges)
    counts, count_ids = get_spike_counts_in_bins(
        spike_times, spike_clusters, intervals
    )
    counts = _reorder_counts(counts, count_ids, unit_ids)
    rates = counts.reshape(unit_ids.size, trial_ids.size, len(edges) - 1) / 0.010
    rates = rates / (baseline_rate[:, None, None] + 1.0)

    usable = (
        np.isfinite(reaction_time)
        & (reaction_time >= 0.080)
        & (reaction_time <= 0.400)
        & np.isin(choice, [-1, 1])
    )
    usable_trials = np.where(usable)[0]
    choice_use = choice[usable_trials]
    contrast_use = signed_contrast[usable_trials]
    rng = _exact_session_rng(f"{eid}_{probe}_claw", claw_seed)
    matched_left, matched_right = _ibl_match_trials_by_signed_contrast_exact(
        choice_use,
        contrast_use,
        n_bins=3,
        minimum_per_bin=5,
        rng=rng,
    )
    trials_left = usable_trials[matched_left]
    trials_right = usable_trials[matched_right]
    trials_all = np.sort(np.concatenate([trials_left, trials_right]))

    population_rates = {
        population: rates[population_rows[population]][:, trials_all, :].mean(
            axis=0
        )
        for population in populations
    }
    right_edges = edges[1:]
    starts = edges[:-1]
    valid_pre = np.flatnonzero(right_edges <= (-0.010 + 1e-12))
    kernel = _exact_causal_kernel(
        0.025 / (0.010 + 1e-12), epsilon=1e-12
    )

    rows: list[dict[str, Any]] = []
    for selected_row, trial in enumerate(trials_all):
        rt = float(reaction_time[trial])
        selected_bins = valid_pre[starts[valid_pre] >= (-rt + 1e-12)]
        if selected_bins.size == 0:
            continue
        smoothed: dict[str, np.ndarray] = {}
        for population in populations:
            segment = population_rates[population][
                selected_row, selected_bins
            ].astype(float)
            smoothed[population] = _exact_causal_smooth(segment, kernel)
        for bin_index, source_bin in enumerate(selected_bins):
            row = {
                "session": result["session"],
                "eid": str(eid),
                "pid": str(pid),
                "trial_id": int(trial_ids[trial]),
                "bin_index": int(bin_index),
                "time_from_stimulus_s": float(starts[source_bin] + rt),
                "time_to_decision_s": float(starts[source_bin]),
                "choice_left": int(choice[trial] == -1),
                "decision_time_ms": float(1000.0 * rt),
            }
            for population in populations:
                row[population] = float(smoothed[population][bin_index])
            rows.append(row)
    return pd.DataFrame(rows)


def _stein_plot_trial_info_exact(
    payload: dict[str, Any],
    matching_seed: int,
) -> dict[str, np.ndarray]:
    """Reproduce the flattened trial logic in plot_clustering_Steinmetz."""
    from .datasets import _steinmetz_match_trials_exact

    session_dir = Path(payload["session_dir"])
    stimulus_times = np.asarray(payload["stimulus_times"], dtype=float).reshape(-1)
    choice = np.asarray(payload["choice"], dtype=int).reshape(-1)
    included = np.asarray(payload["included"], dtype=bool).reshape(-1)
    movement_onsets = np.asarray(payload["movement_onsets"], dtype=float).reshape(-1)
    movement_intervals = np.asarray(payload["movement_intervals"], dtype=float)

    reaction_time = movement_onsets - stimulus_times
    early_movement = np.zeros(len(stimulus_times), dtype=bool)
    starts = movement_intervals[:, 0]
    ends = movement_intervals[:, 1]
    for trial, (window_start, window_end) in enumerate(
        zip(stimulus_times - 0.05, stimulus_times + 0.125)
    ):
        if not (np.isfinite(window_start) and np.isfinite(window_end)):
            early_movement[trial] = True
        else:
            early_movement[trial] = np.any(
                (starts < window_end) & (ends > window_start)
            )

    usable = (
        included
        & np.isfinite(reaction_time)
        & (reaction_time >= 0.125)
        & (reaction_time <= 0.400)
        & (~early_movement)
        & (choice != 0)
    )
    trials_left = np.where(
        usable & (choice == int(payload["choice_left_code"]))
    )[0]
    trials_right = np.where(
        usable & (choice == int(payload["choice_right_code"]))
    )[0]
    trials_left, trials_right, _ = _steinmetz_match_trials_exact(
        session_dir,
        trials_left,
        trials_right,
        included,
        matching_seed=int(matching_seed),
        n_bins=3,
        minimum_per_bin=5,
    )
    return {
        "trials_left": np.asarray(trials_left, dtype=int),
        "trials_right": np.asarray(trials_right, dtype=int),
        "trials_all": np.sort(
            np.concatenate([trials_left, trials_right])
        ).astype(int),
        "reaction_time_s": reaction_time,
    }


def cluster_steinmetz_session_exact(payload: dict[str, Any]) -> dict[str, Any]:

    spike_times = np.asarray(payload["spike_times"], dtype=float)
    spike_clusters = np.asarray(payload["spike_clusters"], dtype=int)
    stimulus_times_raw = np.asarray(payload["stimulus_times_raw"], dtype=float)
    movement_onsets = np.asarray(payload["movement_onsets"], dtype=float)
    choice_raw = np.asarray(payload["choice_raw"], dtype=int)
    included_raw = np.asarray(payload["included_raw"], dtype=bool)
    usable_raw = np.asarray(payload["usable_raw"], dtype=bool)
    trials_left = np.asarray(payload["trials_left"], dtype=int)
    trials_right = np.asarray(payload["trials_right"], dtype=int)
    baseline_trials = included_raw.copy()
    geometry = _stein_geometry_exact(float(payload["premove_gap_s"]))

    retained_units: list[int] = []
    features: list[np.ndarray] = []

    for unit_id in np.asarray(payload["unit_ids_all"], dtype=int):
        unit_spikes = np.sort(spike_times[spike_clusters == unit_id])

        mean_rate = _stein_compute_unit_rate_exact(
            unit_spikes,
            stimulus_times_raw,
            baseline_trials,
            (0.00, 0.40),
        )
        if mean_rate < 0.1:
            continue

        if not _stein_passes_first_modulation_filter_exact(
            unit_spikes,
            stimulus_times_raw,
            movement_onsets,
            baseline_trials,
            trials_left,
            trials_right,
        ):
            continue

        baseline_rate = _stein_compute_unit_rate_exact(
            unit_spikes,
            stimulus_times_raw,
            baseline_trials,
            (-0.20, 0.00),
        )
        normalization = baseline_rate + 1.0

        if not _stein_is_task_modulated_exact(
            unit_spikes,
            stimulus_times_raw,
            movement_onsets,
            choice_raw,
            included_raw,
            usable_raw,
            int(payload["choice_left_code"]),
            int(payload["choice_right_code"]),
        ):
            continue

        left_mean = np.mean(
            np.vstack(
                [
                    _stein_extract_feature_exact(
                        unit_spikes,
                        movement_onsets[trial],
                        normalization,
                        geometry,
                        geometry["stage_a"],
                    )
                    for trial in trials_left
                ]
            ),
            axis=0,
        )
        right_mean = np.mean(
            np.vstack(
                [
                    _stein_extract_feature_exact(
                        unit_spikes,
                        movement_onsets[trial],
                        normalization,
                        geometry,
                        geometry["stage_a"],
                    )
                    for trial in trials_right
                ]
            ),
            axis=0,
        )

        stage_a_times = geometry["centers"][geometry["stage_a"]]
        slope_left = np.polyfit(stage_a_times, left_mean, 1)[0]
        slope_right = np.polyfit(stage_a_times, right_mean, 1)[0]
        if (
            not np.isfinite(slope_left)
            or not np.isfinite(slope_right)
            or slope_left < 1e-4
            or slope_right < 1e-4
        ):
            continue

        feature = np.concatenate([left_mean, right_mean])
        if not np.all(np.isfinite(feature)) or np.ptp(feature) < 0.05:
            continue

        retained_units.append(int(unit_id))
        features.append(feature)

    if len(retained_units) < 8:
        raise ValueError(
            f"{payload['session']}: only {len(retained_units)} units after the original filters."
        )

    unit_ids = np.asarray(retained_units, dtype=int)
    x12 = np.asarray(features, dtype=float)
    x12_z = (x12 - x12.mean(axis=1, keepdims=True)) / (
        x12.std(axis=1, keepdims=True) + _EXACT_EPS
    )

    raw_channel_labels = AgglomerativeClustering(
        n_clusters=2,
        linkage="ward",
        metric="euclidean",
    ).fit_predict(x12_z)
    bias = x12[:, :6].mean(axis=1) - x12[:, 6:].mean(axis=1)
    bias_zero = (
        bias[raw_channel_labels == 0].mean()
        if np.any(raw_channel_labels == 0)
        else -np.inf
    )
    bias_one = (
        bias[raw_channel_labels == 1].mean()
        if np.any(raw_channel_labels == 1)
        else -np.inf
    )
    left_channel_label = 0 if bias_zero > bias_one else 1
    left_rows = np.where(raw_channel_labels == left_channel_label)[0]
    right_rows = np.where(raw_channel_labels != left_channel_label)[0]
    if left_rows.size < 2 or right_rows.size < 2:
        raise ValueError(
            f"{payload['session']}: Stage A produced a group smaller than two units."
        )

    stage_b_trials = np.sort(
        np.concatenate([trials_left, trials_right])
    )
    baseline_rates: dict[int, float] = {}
    for unit_id in unit_ids:
        unit_spikes = np.sort(spike_times[spike_clusters == unit_id])
        baseline_rates[int(unit_id)] = _stein_compute_unit_rate_exact(
            unit_spikes,
            stimulus_times_raw,
            baseline_trials,
            (-0.20, 0.00),
        )

    def build_stage_b_vector(unit_id: int) -> np.ndarray:
        unit_spikes = np.sort(spike_times[spike_clusters == unit_id])
        normalization = baseline_rates[int(unit_id)] + 1.0
        segments = [
            _stein_extract_feature_exact(
                unit_spikes,
                movement_onsets[trial],
                normalization,
                geometry,
                geometry["stage_b"],
            )
            for trial in stage_b_trials
        ]
        vector = np.concatenate(segments)
        return (vector - vector.mean()) / (
            vector.std() + _EXACT_EPS
        )

    stage_b_vectors: list[np.ndarray | None] = [None] * len(unit_ids)
    for row, unit_id in enumerate(unit_ids):
        vector = build_stage_b_vector(int(unit_id))
        if np.all(np.isfinite(vector)) and np.std(vector) >= 1e-6:
            stage_b_vectors[row] = vector

    def split_channel(channel_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        if channel_rows.size < 2 or stage_b_trials.size < 12:
            return None, None
        vectors: list[np.ndarray] = []
        kept: list[int] = []
        for row in channel_rows:
            vector = stage_b_vectors[row]
            if vector is None:
                vector = build_stage_b_vector(int(unit_ids[row]))
            if not np.all(np.isfinite(vector)) or np.std(vector) < 1e-6:
                continue
            vectors.append(vector)
            kept.append(int(row))
        if len(kept) < 2:
            return None, None

        correlation = np.corrcoef(np.vstack(vectors))
        correlation = np.nan_to_num(
            np.clip(correlation, -1.0, 1.0),
            nan=0.0,
        )
        np.fill_diagonal(correlation, 1.0)
        distance = 1.0 - correlation
        np.fill_diagonal(distance, 0.0)

        kept_before = np.asarray(kept, dtype=int)
        kept_after, sublabels = _stein_recluster_exact(
            distance,
            minimum_size=3,
        )
        if kept_after.size < 6:
            return kept_before, np.zeros(
                len(kept_before), dtype=int
            )
        return kept_before[kept_after], sublabels

    kept_left, sublabels_left = split_channel(left_rows)
    kept_right, sublabels_right = split_channel(right_rows)
    if kept_left is None or kept_right is None:
        raise ValueError(
            f"{payload['session']}: Stage B failed in the original pipeline."
        )

    final_labels = np.full(len(unit_ids), -1, dtype=int)
    final_names = np.array(["unassigned"] * len(unit_ids), dtype=object)
    for row, label in zip(kept_left, sublabels_left):
        final_labels[row] = int(label)
        final_names[row] = f"L_sub{label}"
    for row, label in zip(kept_right, sublabels_right):
        final_labels[row] = int(label) + 2
        final_names[row] = f"R_sub{label}"

    left_set = set(left_rows.tolist())
    lr_pref = np.array(
        ["L" if row in left_set else "R" for row in range(len(unit_ids))],
        dtype=object,
    )

    result = {
        "dataset": "Steinmetz",
        "session": payload["session"],
        "session_folder": payload["session_folder"],
        "unit_ids": unit_ids,
        "x12": x12,
        "x12_z": x12_z,
        "lr_pref": lr_pref,
        "raw_channel_labels": raw_channel_labels,
        "left_channel_label": int(left_channel_label),
        "final_labels": final_labels,
        "final_names": final_names,
        "geometry": geometry,
        "source_payload": payload,
    }

    naming_info = _stein_plot_trial_info_exact(
        payload,
        matching_seed=int(payload["matching_seed"]),
    )
    fast_trials, slow_trials, split_metadata = _stein_split_fast_slow_exact(
        naming_info["trials_all"],
        naming_info["reaction_time_s"],
    )
    corr_fast = _exact_corr_matrix(
        _stein_cluster_vectors_exact(result, fast_trials)
    )
    corr_slow = _exact_corr_matrix(
        _stein_cluster_vectors_exact(result, slow_trials)
    )
    delta = corr_slow - corr_fast

    best_pair: tuple[int, int] | None = None
    best_drop = np.inf
    for left_label in (0, 1):
        for right_label in (2, 3):
            drop = delta[left_label, right_label]
            if np.isfinite(drop) and drop < best_drop:
                best_drop = float(drop)
                best_pair = (left_label, right_label)

    name_map: dict[int, str] = {}
    if best_pair is not None:
        d_left, d_right = best_pair
        name_map = {
            int(d_left): "dSPN_left",
            int(1 - d_left): "iSPN_left",
            int(d_right): "dSPN_right",
            int(5 - d_right): "iSPN_right",
        }
        for label, name in name_map.items():
            final_names[final_labels == label] = name

    result.update(
        final_names=final_names,
        name_map=name_map,
        corr_fast=corr_fast,
        corr_slow=corr_slow,
        corr_drop=float(best_drop) if best_pair is not None else np.nan,
        fast_trials=fast_trials,
        slow_trials=slow_trials,
        split_metadata=split_metadata,
        naming_trial_info=naming_info,
    )
    return result


def _unit_profile_rows_from_x12_exact(
    result: dict[str, Any],
    x12_z: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for unit_row, unit_id in enumerate(result["unit_ids"]):
        if int(result["final_labels"][unit_row]) < 0:
            continue
        for feature_index, value in enumerate(x12_z[unit_row]):
            rows.append(
                {
                    "session": result["session"],
                    "unit_id": int(unit_id),
                    "spn_name": str(result["final_names"][unit_row]),
                    "choice_side": "left" if feature_index < 6 else "right",
                    "bin_from_decision": int(-6 + (feature_index % 6)),
                    "z_rate": float(value),
                }
            )
    return pd.DataFrame(rows)


def _correlation_rows_exact(
    result: dict[str, Any],
    corr_fast: np.ndarray,
    corr_slow: np.ndarray,
    fast_trials: np.ndarray,
    slow_trials: np.ndarray,
    split_metadata: dict[str, float],
) -> pd.DataFrame:
    name_map = _label_name_map(
        result["final_labels"], result["final_names"]
    )
    metadata = {
        "fast": (
            len(fast_trials),
            1000.0 * float(split_metadata.get("fast_min_s", np.nan)),
        ),
        "slow": (
            len(slow_trials),
            1000.0 * float(split_metadata.get("slow_min_s", np.nan)),
        ),
    }
    rows: list[dict[str, Any]] = []
    for speed, matrix in (("fast", corr_fast), ("slow", corr_slow)):
        n_trials, cutoff = metadata[speed]
        for row_label in range(4):
            for col_label in range(4):
                rows.append(
                    {
                        "session": result["session"],
                        "speed": speed,
                        "n_trials": int(n_trials),
                        "decision_time_cutoff_ms": float(cutoff),
                        "spn_i": name_map[row_label],
                        "spn_j": name_map[col_label],
                        "pearson_r": float(matrix[row_label, col_label]),
                    }
                )
    return pd.DataFrame(rows)


def steinmetz_plot_tables_exact(
    result: dict[str, Any],
    plot_matching_seed: int = 27,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Rebuild manuscript profiles/correlations as in the plotting notebook."""
    payload = result["source_payload"]
    info = _stein_plot_trial_info_exact(
        payload,
        matching_seed=int(plot_matching_seed),
    )
    spike_times = np.asarray(payload["spike_times"], dtype=float)
    spike_clusters = np.asarray(payload["spike_clusters"], dtype=int)
    stimulus_times = np.asarray(payload["stimulus_times"], dtype=float)
    movement_onsets = np.asarray(payload["movement_onsets"], dtype=float)
    baseline_trials = np.asarray(payload["included"], dtype=bool)
    geometry = _stein_geometry_exact(float(payload["premove_gap_s"]))

    rebuilt = []
    for unit_id in np.asarray(result["unit_ids"], dtype=int):
        unit_spikes = np.sort(spike_times[spike_clusters == unit_id])
        baseline_rate = _stein_compute_unit_rate_exact(
            unit_spikes,
            stimulus_times,
            baseline_trials,
            (-0.20, 0.00),
        )
        normalization = baseline_rate + 1.0
        left_mean = np.mean(
            np.vstack(
                [
                    _stein_extract_feature_exact(
                        unit_spikes,
                        movement_onsets[trial],
                        normalization,
                        geometry,
                        geometry["stage_a"],
                    )
                    for trial in info["trials_left"]
                ]
            ),
            axis=0,
        )
        right_mean = np.mean(
            np.vstack(
                [
                    _stein_extract_feature_exact(
                        unit_spikes,
                        movement_onsets[trial],
                        normalization,
                        geometry,
                        geometry["stage_a"],
                    )
                    for trial in info["trials_right"]
                ]
            ),
            axis=0,
        )
        rebuilt.append(np.concatenate([left_mean, right_mean]))

    x12 = np.asarray(rebuilt, dtype=float)
    x12_z = (x12 - x12.mean(axis=1, keepdims=True)) / (
        x12.std(axis=1, keepdims=True) + _EXACT_EPS
    )
    profile_table = _unit_profile_rows_from_x12_exact(result, x12_z)

    fast_trials, slow_trials, split_metadata = _stein_split_fast_slow_exact(
        info["trials_all"], info["reaction_time_s"]
    )
    corr_fast = _exact_corr_matrix(
        _stein_cluster_vectors_exact(result, fast_trials)
    )
    corr_slow = _exact_corr_matrix(
        _stein_cluster_vectors_exact(result, slow_trials)
    )
    correlation_table = _correlation_rows_exact(
        result,
        corr_fast,
        corr_slow,
        fast_trials,
        slow_trials,
        split_metadata,
    )
    metadata = {
        "plot_matching_seed": int(plot_matching_seed),
        "n_plot_trials": int(len(info["trials_all"])),
        "n_plot_left_trials": int(len(info["trials_left"])),
        "n_plot_right_trials": int(len(info["trials_right"])),
    }
    return profile_table, correlation_table, metadata


def ibl_plot_tables_exact(
    result: dict[str, Any],
    plot_matching_seed: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Rebuild IBL manuscript profiles/correlations as in the plotting notebook."""
    from .datasets import (
        _exact_session_rng,
        _ibl_match_trials_by_signed_contrast_exact,
    )

    choice = np.asarray(result["choice"], dtype=int)
    signed_contrast = np.asarray(result["signed_contrast"], dtype=float)
    rng = _exact_session_rng(
        f"{result['eid']}_{result['probe']}",
        int(plot_matching_seed),
    )
    idx_left, idx_right = _ibl_match_trials_by_signed_contrast_exact(
        choice,
        signed_contrast,
        n_bins=3,
        minimum_per_bin=10,
        rng=rng,
    )

    rates = np.asarray(result["rates"], dtype=float)
    stage_a = np.asarray(result["stage_a_bins"], dtype=int)
    left_mean = rates[:, idx_left, :][:, :, stage_a].mean(axis=1)
    right_mean = rates[:, idx_right, :][:, :, stage_a].mean(axis=1)
    x12 = np.concatenate([left_mean, right_mean], axis=1)
    x12_z = (x12 - x12.mean(axis=1, keepdims=True)) / (
        x12.std(axis=1, keepdims=True) + _EXACT_EPS
    )
    profile_table = _unit_profile_rows_from_x12_exact(result, x12_z)

    trials_all = np.sort(np.concatenate([idx_left, idx_right]))
    fast_trials, slow_trials, split_metadata = _ibl_split_fast_slow_exact(
        trials_all,
        np.asarray(result["reaction_time_s"], dtype=float),
    )
    corr_fast = _exact_corr_matrix(
        _ibl_cluster_vectors_exact(result, fast_trials)
    )
    corr_slow = _exact_corr_matrix(
        _ibl_cluster_vectors_exact(result, slow_trials)
    )
    correlation_table = _correlation_rows_exact(
        result,
        corr_fast,
        corr_slow,
        fast_trials,
        slow_trials,
        split_metadata,
    )
    metadata = {
        "plot_matching_seed": int(plot_matching_seed),
        "n_plot_trials": int(len(trials_all)),
        "n_plot_left_trials": int(len(idx_left)),
        "n_plot_right_trials": int(len(idx_right)),
    }
    return profile_table, correlation_table, metadata