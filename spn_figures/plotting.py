from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import SPN_COLORS, SPN_LINESTYLES, SPN_NAMES


def plot_cbgt_profiles(feature_table: pd.DataFrame):
    feature_cols = [f"feature_{i:02d}" for i in range(12)]
    x = np.arange(12)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for name in ["dSPN_left", "iSPN_left", "dSPN_right", "iSPN_right"]:
        values = feature_table.loc[feature_table["true_name"] == name, feature_cols].to_numpy(float)
        values = (values - values.mean(axis=1, keepdims=True)) / (values.std(axis=1, keepdims=True) + 1e-9)
        mean = values.mean(axis=0)
        sem = values.std(axis=0) / np.sqrt(len(values))
        ax.plot(x[:6], mean[:6], marker="o", color=SPN_COLORS[name], linestyle=SPN_LINESTYLES[name], label=name)
        ax.plot(x[6:], mean[6:], marker="o", color=SPN_COLORS[name], linestyle=SPN_LINESTYLES[name])
        ax.fill_between(x[:6], mean[:6] - sem[:6], mean[:6] + sem[:6], color=SPN_COLORS[name], alpha=0.12)
        ax.fill_between(x[6:], mean[6:] - sem[6:], mean[6:] + sem[6:], color=SPN_COLORS[name], alpha=0.12)
    ax.axvline(5.5, color="0.5", linestyle=":")
    ax.set_xticks(x)
    #ax.set_xticklabels([f"L-{i}" for i in range(6, 0, -1)] + [f"R-{i}" for i in range(6, 0, -1)])
    ax.set_xticklabels([-60, -50, -40, -30, -20, -10, -60, -50, -40, -30, -20, -10], fontsize=12)
    ax.set_ylabel("Z-scored firing rate")
    ax.set_xlabel("Bins before DT (ms), Left | Right", fontsize=12.4)
    #ax.set_title("12D Firing rate profiles")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5,1.2), frameon=False)
    return fig, ax


def cbgt_speed_correlations(activity: pd.DataFrame) -> dict[str, np.ndarray]:
    matrices = {}
    for speed in ("fast", "slow"):
        data = activity[activity["speed_group"].str.lower() == speed]
        vectors = []
        for name in SPN_NAMES:
            network_vectors = []
            for _network, group in data[data["spn_name"] == name].groupby("network_id"):
                vector = group.sort_values(["trial_id", "bin_from_decision"])["rate"].to_numpy(float)
                network_vectors.append((vector - vector.mean()) / (vector.std() + 1e-9))
            vectors.append(np.concatenate(network_vectors))
        matrices[speed] = np.corrcoef(np.vstack(vectors))
    return matrices


def plot_correlation_matrices(matrices: dict[str, np.ndarray], titles=("Fast", "Slow")):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), constrained_layout=True)
    image = None
    for ax, key, title in zip(axes, ("fast", "slow"), titles):
        matrix = matrices[key]
        image = ax.imshow(matrix, vmin=-0.3, vmax=1.0, cmap="plasma_r")
        ax.set_title(title)
        ax.set_xticks(range(4), SPN_NAMES, rotation=25, ha="right", fontsize=10)
        ax.set_yticks(range(4), SPN_NAMES if ax is axes[0] else ["", "", "", ""], fontsize=10)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white", fontsize=10)
    fig.colorbar(image, ax=axes, label="Pearson r", shrink=0.8)
    return fig, axes


def plot_clustering_validation(feature_table: pd.DataFrame, vector_table: pd.DataFrame, clustering_result: dict):
    """Plot the Stage 1 and four-subtype confusion matrices."""
    del feature_table, vector_table
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
    # Stage 1 confusion matrix
    matrix = clustering_result["stage1_confusion"]
    image = axes[0].imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center")
    axes[0].set_xticks([0, 1], ["SPN_left", "SPN_right"])
    axes[0].set_yticks([0, 1], ["SPN_left", "SPN_right"])
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_title(f"Stage 1 confusion matrix\n"
                      f"ARI={clustering_result['stage1_ari']:.3f}")
    # Four-subtype confusion matrix
    matrix = clustering_result["stage2_confusion"]
    axes[1].imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    for i in range(4):
        for j in range(4):
            axes[1].text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
    labels = ["dL", "iL", "dR", "iR"]
    axes[1].set_xticks(range(4), labels)
    axes[1].set_yticks(range(4), labels)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].set_title(f"Four-subtype matrix\n" 
                      f"ARI={clustering_result['stage2_ari']:.3f}")
    fig.colorbar(image, ax=axes, shrink=0.8, label="Pearson r")
    return fig


def _profile_summary(profile: pd.DataFrame, session: str) -> pd.DataFrame:
    data = profile[profile["session"] == session].copy()
    data["window"] = pd.cut(
        data["bin_from_decision"],
        bins=[-7, -5, -3, -1],
        labels=["Early", "Mid", "Late"],
        include_lowest=True,
    )

    # Average the two bins within each unit first, then calculate the
    # population mean and SEM across units.
    unit_windows = (
        data.groupby(
            ["unit_id", "spn_name", "choice_side", "window"],
            observed=True,
        )["z_rate"]
        .mean()
        .reset_index()
    )
    return (
        unit_windows.groupby(
            ["spn_name", "choice_side", "window"],
            observed=True,
        )["z_rate"]
        .agg(["mean", "sem"])
        .reset_index()
    )


def plot_session_profile(ax, profile: pd.DataFrame, session: str, n_trials: int | None = None):
    summary = _profile_summary(profile, session)
    positions = {("left", "Early"): 0, ("left", "Mid"): 1, ("left", "Late"): 2,
                 ("right", "Early"): 4, ("right", "Mid"): 5, ("right", "Late"): 6}
    for name in ["dSPN_left", "iSPN_left", "dSPN_right", "iSPN_right"]:
        group = summary[summary["spn_name"] == name]
        for side in ("left", "right"):
            part = group[group["choice_side"] == side]
            x = [positions[(side, window)] for window in ("Early", "Mid", "Late")]
            y = [part.loc[part["window"] == window, "mean"].iloc[0] for window in ("Early", "Mid", "Late")]
            e = [part.loc[part["window"] == window, "sem"].iloc[0] for window in ("Early", "Mid", "Late")]
            ax.errorbar(
                x, y, yerr=e, marker="o", color=SPN_COLORS[name],
                linestyle=SPN_LINESTYLES[name], linewidth=1.5,
                label=name if side == "left" else "_nolegend_",
            )
    counts = (
        profile[profile["session"] == session]
        .drop_duplicates(["unit_id", "spn_name"])["spn_name"]
        .value_counts()
    )
    abbreviations = {"dSPN_left": "dL", "iSPN_left": "iL", "dSPN_right": "dR", "iSPN_right": "iR"}
    count_text = ", ".join(f"{abbreviations[name]}={int(counts.get(name, 0))}" for name in ["dSPN_left", "iSPN_left", "dSPN_right", "iSPN_right"])
    ax.text(0.02, 0.95, count_text, transform=ax.transAxes, va="top", color="0.55", fontsize=8)
    if n_trials is not None:
        ax.text(0.98, 0.05, f"{int(n_trials)} trials", transform=ax.transAxes, ha="right", va="bottom", color="0.55", fontsize=8)
    ax.axvline(3, color="0.4", linestyle=":")
    ax.set_xticks(
        [0, 1, 2, 4, 5, 6],
        [
            "Early\n(L-6..L-5)",
            "Mid\n(L-4..L-3)",
            "Late\n(L-2..L-1)",
            "Early\n(R-6..R-5)",
            "Mid\n(R-4..R-3)",
            "Late\n(R-2..R-1)",
        ],
    )
    ax.set_ylabel("Z-scored firing rate")
    ax.set_title(session, fontsize=10)


def _corr_matrix(corr: pd.DataFrame, session: str, speed: str) -> np.ndarray:
    data = corr[(corr["session"] == session) & (corr["speed"] == speed)]
    return data.pivot(index="spn_i", columns="spn_j", values="pearson_r").reindex(index=SPN_NAMES, columns=SPN_NAMES).to_numpy(float)


def plot_empirical_sessions(profile: pd.DataFrame, corr: pd.DataFrame, sessions: list[str], title: str | None = None):
    fig = plt.figure(figsize=(10.2, 2.75 * len(sessions) + 0.45))
    grid = fig.add_gridspec(
        len(sessions), 3,
        width_ratios=[1.35, 1, 1],
        left=0.07, right=0.90,
        bottom=0.07, top=0.90,
        hspace=0.34, wspace=0.24,
    )
    image = None
    first_profile_ax = None
    for row, session in enumerate(sessions):
        profile_ax = fig.add_subplot(grid[row, 0])
        if first_profile_ax is None:
            first_profile_ax = profile_ax
        session_corr = corr[corr["session"] == session]
        n_trials = int(session_corr.groupby("speed")["n_trials"].first().sum())
        plot_session_profile(profile_ax, profile, session, n_trials=n_trials)
        for col, speed in enumerate(("fast", "slow"), start=1):
            ax = fig.add_subplot(grid[row, col])
            matrix = _corr_matrix(corr, session, speed)
            image = ax.imshow(matrix, vmin=-0.3, vmax=1.0, cmap="plasma_r")
            metadata = session_corr[session_corr["speed"] == speed].iloc[0]
            inequality = "≤" if speed == "fast" else "≥"
            ax.set_title(
                f"{speed.capitalize()} (#trials={int(metadata['n_trials'])}; "
                f"RT{inequality}{metadata['decision_time_cutoff_ms']:.0f}ms)",
                fontsize=8,
                pad=4,
            )
            ax.set_xticks(range(4), ["dL", "dR", "iL", "iR"], fontsize=7)
            ax.set_yticks(range(4), ["dL", "dR", "iL", "iR"] if col == 1 else ["", "", "", ""], fontsize=7)
            for i in range(4):
                for j in range(4):
                    ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white", fontsize=6)

    handles, labels = first_profile_ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.55, 0.985), frameon=False, fontsize=8)
    if title:
        fig.text(0.02, 0.985, title, ha="left", va="top", fontsize=11, fontweight="bold")
    colorbar_ax = fig.add_axes([0.925, 0.20, 0.014, 0.58])
    fig.colorbar(image, cax=colorbar_ax, label="Pearson r")
    return fig


def draw_percentile_box(ax, x, q, facecolor, width=0.18):
    q02, q25, q50, q75, q97 = q
    ax.vlines(x, q02, q97, color="0.2", linewidth=1)
    ax.hlines([q02, q97], x - width * 0.25, x + width * 0.25, color="0.2", linewidth=1)
    ax.add_patch(plt.Rectangle((x - width / 2, q25), width, q75 - q25, facecolor=facecolor, edgecolor="0.2"))
    ax.hlines(q50, x - width / 2, x + width / 2, color="0.1", linewidth=1.4)


def _stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def _bracket(ax, x1, x2, y, text, color="0.15"):
    height = 0.025 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y], color=color, linewidth=0.9)
    ax.text((x1 + x2) / 2, y + height, text, ha="center", va="bottom", color=color, fontsize=8)


def plot_prediction_boxplots(summary: pd.DataFrame, tests: pd.DataFrame):
    fig, left_axes = plt.subplots(1, 3, figsize=(12.8, 4.1))
    right_axes = [ax.twinx() for ax in left_axes]

    cbgt_color = "#D9D9D9"
    empirical_color = "#56B4E9"
    empirical_axis_color = "#2B8CBE"
    offset = {"CBGT": -0.13, "Empirical": 0.13}

    panel_groups = {
        "A": ["iSPN-only", "Left d(+i)", "Right d(+i)"],
        "B": ["Left d", "Left d(+i)", "Right d", "Right d(+i)"],
        "C": ["Left d(+i)", "Left d(+i)\n+ Right i", "Right d(+i)", "Right d(+i)\n+ Left i"],
    }
    titles = {
        "A": "iSPN-only states preserve uncertainty",
        "B": "Same-channel d+i marks terminality",
        "C": "Slower decisions with opponent iSPN",
    }
    ylabels = {
        "A": "Left-choice probability",
        "B": "Terminal probability",
        "C": "Decision time (ms)",
    }

    axis_for = {
        "CBGT": dict(axes=left_axes, color=cbgt_color),
        "Empirical": dict(axes=right_axes, color=empirical_color),
    }

    def set_limits(ax, rows: pd.DataFrame, panel: str) -> None:
        low = float(rows["q02_5"].min())
        high = float(rows["q97_5"].max())
        if panel in ("A", "B"):
            if panel == "A":
                low = min(low, 0.5)
                high = max(high, 0.5)
            span = max(high - low, 0.15)
            lower = max(0.0, low - 0.12 * span)
            upper = min(1.0, high + 0.32 * span)
            if upper - lower < 0.25:
                upper = min(1.0, lower + 0.25)
            ax.set_ylim(lower, upper)
        else:
            span = max(high - low, 1.0)
            ax.set_ylim(low - 0.10 * span, high + 0.30 * span)

    for panel_index, panel in enumerate(("A", "B", "C")):
        left_ax = left_axes[panel_index]
        right_ax = right_axes[panel_index]
        groups = panel_groups[panel]

        for dataset in ("CBGT", "Empirical"):
            axis = axis_for[dataset]["axes"][panel_index]
            color = axis_for[dataset]["color"]
            panel_rows = summary[(summary["dataset"] == dataset) & (summary["panel"] == panel)]
            set_limits(axis, panel_rows, panel)
            for group_index, group in enumerate(groups):
                row = panel_rows[panel_rows["group"] == group].iloc[0]
                quantiles = row[["q02_5", "q25", "q50", "q75", "q97_5"]].to_numpy(float)
                draw_percentile_box(axis, group_index + offset[dataset], quantiles, color)

        left_ax.set_xticks(range(len(groups)), groups)
        left_ax.set_title(titles[panel], fontsize=10.5)
        left_ax.set_ylabel(f"CBGT {ylabels[panel]}")
        right_ax.set_ylabel(f"Empirical {ylabels[panel]}", color=empirical_axis_color)
        left_ax.tick_params(axis="x", labelsize=8)
        right_ax.tick_params(axis="y", colors=empirical_axis_color)
        right_ax.spines["right"].set_color(empirical_axis_color)
        right_ax.patch.set_visible(False)

        if panel == "A":
            left_ax.axhline(0.5, color="0.55", linestyle="--", linewidth=0.9)
            right_ax.axhline(0.5, color=empirical_axis_color, linestyle="--", linewidth=0.8, alpha=0.7)
        if panel in ("B", "C"):
            left_ax.axvline(1.5, color="0.75", linestyle=":")

    comparisons_by_panel = {
        "A": [(0, 1, "iSPN-only vs Left d(+i)"), (0, 2, "iSPN-only vs Right d(+i)")],
        "B": [(0, 1, "Left d vs Left d(+i)"), (2, 3, "Right d vs Right d(+i)")],
        "C": [(0, 1, "Left"), (2, 3, "Right")],
    }
    for dataset, line_color in (("CBGT", "0.15"), ("Empirical", empirical_axis_color)):
        for panel_index, panel in enumerate(("A", "B", "C")):
            ax = axis_for[dataset]["axes"][panel_index]
            low, high = ax.get_ylim()
            span = high - low
            for level, (i, j, comparison) in enumerate(comparisons_by_panel[panel]):
                row = tests[
                    (tests["dataset"] == dataset)
                    & (tests["panel"] == panel)
                    & (tests["comparison"] == comparison)
                ]
                if len(row):
                    base = 0.08 if dataset == "CBGT" else 0.20
                    y = high - (base + 0.13 * level) * span
                    _bracket(
                        ax,
                        i + offset[dataset],
                        j + offset[dataset],
                        y,
                        _stars(float(row["p_value"].iloc[0])),
                        line_color,
                    )

    handles = [
        mpl.patches.Patch(facecolor=cbgt_color, edgecolor="0.2", label="CBGT"),
        mpl.patches.Patch(facecolor=empirical_color, edgecolor="0.2", label="IBL + Steinmetz"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False)
    fig.subplots_adjust(top=0.79, bottom=0.19, wspace=0.58)
    return fig


_EMPIRICAL_PROFILE_ORDER = [
    "dSPN_left",
    "iSPN_left",
    "dSPN_right",
    "iSPN_right",
]
_EMPIRICAL_CORR_ORDER = [
    "dSPN_left",
    "dSPN_right",
    "iSPN_left",
    "iSPN_right",
]
_EMPIRICAL_OFFSETS = [0.00, 0.22, 0.55, 0.77]
_EMPIRICAL_GROUP_STEP = 1.60
_EMPIRICAL_GROUP_CENTERS = [
    0.60,
    2.20,
    3.80,
    5.40,
    7.00,
    8.60,
]
_EMPIRICAL_XTICKLABELS = [
    "Early\n(L-6..L-5)",
    "Mid\n(L-4..L-3)",
    "Late\n(L-2..L-1)",
    "Early\n(R-6..R-5)",
    "Mid\n(R-4..R-3)",
    "Late\n(R-2..R-1)",
]


def _empirical_session_matrix(
    profile: pd.DataFrame,
    session: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return row-z-scored 12D profiles, unit IDs, and subtype names."""
    data = profile.loc[profile["session"] == session].copy()
    if data.empty:
        raise KeyError(f"No profile rows found for session {session!r}.")

    expected_bins = list(range(-6, 0))
    rows: list[np.ndarray] = []
    unit_ids: list[int] = []
    names: list[str] = []
    for (unit_id, name), group in data.groupby(
        ["unit_id", "spn_name"], sort=True
    ):
        values: list[float] = []
        for side in ("left", "right"):
            part = (
                group.loc[group["choice_side"] == side]
                .set_index("bin_from_decision")["z_rate"]
                .reindex(expected_bins)
            )
            if part.isna().any():
                raise ValueError(
                    f"{session}: incomplete 12D profile for unit {unit_id}, {name}."
                )
            values.extend(part.to_numpy(float).tolist())
        rows.append(np.asarray(values, dtype=float))
        unit_ids.append(int(unit_id))
        names.append(str(name))

    return (
        np.vstack(rows),
        np.asarray(unit_ids, dtype=int),
        np.asarray(names, dtype=object),
    )


def _empirical_profile_ylim(
    x12: np.ndarray,
    names: np.ndarray,
    pad: float = 0.05,
) -> tuple[float, float]:
    """Match the original profile-limit calculation using population mean +/- SE."""
    minima: list[float] = []
    maxima: list[float] = []
    for name in np.unique(names):
        values = x12[names == name]
        if values.size == 0:
            continue
        mean = values.mean(axis=0)
        sem = values.std(axis=0, ddof=0) / np.sqrt(len(values))
        minima.append(float(np.min(mean - sem)))
        maxima.append(float(np.max(mean + sem)))

    ymin = min(minima) if minima else float(np.min(x12))
    ymax = max(maxima) if maxima else float(np.max(x12))
    span = ymax - ymin
    if not np.isfinite(span) or span <= 0:
        span = 1.0
    return ymin - pad * span, ymax + pad * span


def _empirical_trial_count(corr: pd.DataFrame, session: str) -> int:
    rows = corr.loc[corr["session"] == session]
    if rows.empty:
        raise KeyError(f"No correlation rows found for session {session!r}.")
    return int(rows.groupby("speed")["n_trials"].first().sum())


def _plot_empirical_profile_axis_exact(
    ax,
    profile: pd.DataFrame,
    corr: pd.DataFrame,
    session: str,
    *,
    show_xlabels: bool,
    show_ylabel: bool = True,
    title_fontsize: float = 9.0,
    tick_fontsize: float = 7.5,
    annotation_fontsize: float = 7.5,
    line_width: float = 1.7,
    marker_size: float = 4.0,
) -> None:
    """Draw one Early/Mid/Late profile panel using the original plot semantics."""
    x12, _unit_ids, names = _empirical_session_matrix(profile, session)
    name_set = set(map(str, names))
    missing = [name for name in _EMPIRICAL_PROFILE_ORDER if name not in name_set]
    if missing:
        raise ValueError(f"{session}: missing SPN subtype(s): {missing}")

    ymin, ymax = _empirical_profile_ylim(x12, names, pad=0.05)
    line_points: dict[tuple[str, str], dict[str, list[float] | str]] = {}
    segments = ((0, 1), (2, 3), (4, 5))
    base_right = 3 * _EMPIRICAL_GROUP_STEP

    for side in ("left", "right"):
        column_offset = 0 if side == "left" else 6
        base_shift = 0.0 if side == "left" else base_right
        for group_index, bins in enumerate(segments):
            base = base_shift + group_index * _EMPIRICAL_GROUP_STEP
            for subtype_index, name in enumerate(_EMPIRICAL_PROFILE_ORDER):
                values = x12[names == name][:, [column_offset + b for b in bins]].mean(
                    axis=1
                )
                mean = float(np.mean(values))
                sem = float(np.std(values, ddof=0) / np.sqrt(len(values)))
                xpos = base + _EMPIRICAL_OFFSETS[subtype_index]
                color = SPN_COLORS[name]
                ax.errorbar(
                    xpos,
                    mean,
                    yerr=sem,
                    fmt="o",
                    color=color,
                    markersize=max(2.5, marker_size * 0.65),
                    capsize=1.7,
                    capthick=0.8,
                    elinewidth=0.8,
                    zorder=4,
                )
                key = (side, name)
                if key not in line_points:
                    line_points[key] = {"x": [], "y": [], "color": color}
                line_points[key]["x"].append(float(xpos))
                line_points[key]["y"].append(mean)

    for (_side, name), points in line_points.items():
        ax.plot(
            points["x"],
            points["y"],
            color=points["color"],
            linestyle=SPN_LINESTYLES[name],
            linewidth=line_width,
            marker="o",
            markersize=marker_size,
            markeredgewidth=0,
            alpha=0.98,
            zorder=5,
        )

    last_left = 2 * _EMPIRICAL_GROUP_STEP + _EMPIRICAL_OFFSETS[-1]
    first_right = base_right + _EMPIRICAL_OFFSETS[0]
    ax.axvline(
        0.5 * (last_left + first_right),
        linestyle=":",
        linewidth=0.9,
        color="0.25",
        alpha=0.75,
    )

    ax.set_xlim(-0.50, base_right + 2 * _EMPIRICAL_GROUP_STEP + 1.27)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(_EMPIRICAL_GROUP_CENTERS)
    if show_xlabels:
        ax.set_xticklabels(_EMPIRICAL_XTICKLABELS, fontsize=tick_fontsize)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", labelsize=tick_fontsize, pad=1.5)
    ax.set_ylabel(
        "Z-scored firing rate" if show_ylabel else "",
        fontsize=tick_fontsize + 1.0,
        labelpad=2.0,
    )
    ax.set_title(session, fontsize=title_fontsize, pad=2.0)

    counts = {
        name: int(np.sum(names == name)) for name in _EMPIRICAL_PROFILE_ORDER
    }
    count_text = (
        f"dL={counts['dSPN_left']}, iL={counts['iSPN_left']}, "
        f"dR={counts['dSPN_right']}, iR={counts['iSPN_right']}"
    )
    ax.text(
        0.02,
        0.95,
        count_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=annotation_fontsize,
        color="0.58",
    )
    ax.text(
        0.985,
        0.035,
        f"{_empirical_trial_count(corr, session)} trials",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=annotation_fontsize,
        color="0.58",
    )


def _empirical_corr_matrix_exact(
    corr: pd.DataFrame,
    session: str,
    speed: str,
) -> tuple[np.ndarray, pd.Series]:
    rows = corr.loc[
        (corr["session"] == session) & (corr["speed"].str.lower() == speed.lower())
    ]
    if len(rows) != 16:
        raise ValueError(
            f"{session}, {speed}: expected 16 correlation rows, found {len(rows)}."
        )
    matrix = (
        rows.pivot(index="spn_i", columns="spn_j", values="pearson_r")
        .reindex(index=_EMPIRICAL_CORR_ORDER, columns=_EMPIRICAL_CORR_ORDER)
        .to_numpy(float)
    )
    return matrix, rows.iloc[0]


def _plot_empirical_corr_axis_exact(
    ax,
    corr: pd.DataFrame,
    session: str,
    speed: str,
    *,
    show_xlabels: bool,
    show_ylabels: bool,
    title_fontsize: float = 7.0,
    tick_fontsize: float = 6.0,
    value_fontsize: float = 6.0,
):
    matrix, metadata = _empirical_corr_matrix_exact(corr, session, speed)
    cmap = plt.get_cmap("plasma_r").copy()
    cmap.set_bad(color="lightgray")
    image = ax.imshow(
        np.ma.masked_invalid(matrix),
        vmin=-0.35,
        vmax=1.0,
        cmap=cmap,
        interpolation="nearest",
        aspect="equal",
    )
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    if show_xlabels:
        ax.set_xticklabels(
            _EMPIRICAL_CORR_ORDER,
            rotation=25,
            ha="right",
            fontsize=tick_fontsize,
        )
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
    ax.set_yticklabels(
        _EMPIRICAL_CORR_ORDER if show_ylabels else ["", "", "", ""],
        fontsize=tick_fontsize,
    )
    ax.tick_params(axis="both", length=0, pad=1.2)

    speed_title = speed.capitalize()
    ax.set_title(
        f"{speed_title} (#trials={int(metadata['n_trials'])}; "
        f"RT≥{float(metadata['decision_time_cutoff_ms']):.0f}ms)",
        fontsize=title_fontsize,
        pad=2.5,
    )
    for row in range(4):
        for col in range(4):
            value = matrix[row, col]
            text = "nan" if not np.isfinite(value) else f"{value:.2f}"
            ax.text(
                col,
                row,
                text,
                ha="center",
                va="center",
                color="white",
                fontsize=value_fontsize,
            )
    return image


def empirical_profile_legend_handles():
    from matplotlib.lines import Line2D

    return [
        Line2D(
            [0],
            [0],
            color=SPN_COLORS[name],
            linestyle=SPN_LINESTYLES[name],
            marker="o",
            markersize=4,
            linewidth=1.5,
            label=name,
        )
        for name in _EMPIRICAL_PROFILE_ORDER
    ]


def plot_empirical_example_profiles_exact(
    profile: pd.DataFrame,
    corr: pd.DataFrame,
    sessions: list[str],
):
    
    fig = plt.figure(figsize=(5.45, 7.80))
    grid = fig.add_gridspec(
        3,
        1,
        left=0.12,
        right=0.99,
        bottom=0.105,
        top=0.91,
        hspace=0.18,
    )
    axes = []
    for row, session in enumerate(sessions):
        ax = fig.add_subplot(grid[row, 0])
        _plot_empirical_profile_axis_exact(
            ax,
            profile,
            corr,
            session,
            show_xlabels=(row == len(sessions) - 1),
            show_ylabel=True,
            title_fontsize=9.2,
            tick_fontsize=7.4,
            annotation_fontsize=7.2,
            line_width=1.55,
            marker_size=3.8,
        )
        axes.append(ax)

    handles = empirical_profile_legend_handles()
    fig.legend(
        handles=handles,
        labels=[handle.get_label() for handle in handles],
        loc="upper center",
        bbox_to_anchor=(0.56, 0.987),
        ncol=4,
        frameon=False,
        fontsize=8.2,
        columnspacing=1.3,
        handlelength=2.0,
        handletextpad=0.35,
    )
    return fig, axes


def plot_empirical_profile_correlation_grid_exact(
    profile: pd.DataFrame,
    corr: pd.DataFrame,
    sessions: list[str],
    *,
    dataset: str,
):
    dataset_key = dataset.strip().lower()
    n_rows = len(sessions)
    if dataset_key == "steinmetz":
        figsize = (9.05, 8.50)
        bottom, top, hspace = 0.105, 0.985, 0.20
        profile_title_size = 8.4
        corr_title_size = 6.4
        tick_size = 5.7
        annotation_size = 6.4
    else:
        figsize = (9.05, 13.10)
        bottom, top, hspace = 0.070, 0.992, 0.14
        profile_title_size = 7.8
        corr_title_size = 5.9
        tick_size = 5.2
        annotation_size = 5.8

    fig = plt.figure(figsize=figsize)
    grid = fig.add_gridspec(
        n_rows,
        4,
        width_ratios=[1.90, 1.00, 1.00, 0.055],
        left=0.075,
        right=0.985,
        bottom=bottom,
        top=top,
        hspace=hspace,
        wspace=0.19,
    )

    for row, session in enumerate(sessions):
        show_x = row == n_rows - 1
        profile_ax = fig.add_subplot(grid[row, 0])
        _plot_empirical_profile_axis_exact(
            profile_ax,
            profile,
            corr,
            session,
            show_xlabels=show_x,
            show_ylabel=True,
            title_fontsize=profile_title_size,
            tick_fontsize=tick_size,
            annotation_fontsize=annotation_size,
            line_width=1.35 if dataset_key == "ibl" else 1.45,
            marker_size=3.2 if dataset_key == "ibl" else 3.5,
        )

        fast_ax = fig.add_subplot(grid[row, 1])
        fast_image = _plot_empirical_corr_axis_exact(
            fast_ax,
            corr,
            session,
            "fast",
            show_xlabels=show_x,
            show_ylabels=True,
            title_fontsize=corr_title_size,
            tick_fontsize=tick_size,
            value_fontsize=tick_size,
        )

        slow_ax = fig.add_subplot(grid[row, 2])
        _plot_empirical_corr_axis_exact(
            slow_ax,
            corr,
            session,
            "slow",
            show_xlabels=show_x,
            show_ylabels=False,
            title_fontsize=corr_title_size,
            tick_fontsize=tick_size,
            value_fontsize=tick_size,
        )

        colorbar_ax = fig.add_subplot(grid[row, 3])
        colorbar = fig.colorbar(fast_image, cax=colorbar_ax)
        colorbar.ax.tick_params(labelsize=max(4.5, tick_size - 0.6), length=1.5, pad=1)
        colorbar.set_label("Pearson r", fontsize=tick_size, labelpad=2.0)

    return fig


_FIG5_CBGT_BOX_FACE = "#D9D9D9"
_FIG5_EMPIRICAL_BOX_FACE = "#56B4E9"
_FIG5_BOX_EDGE = "0.15"
_FIG5_BOX_ALPHA = 0.95
_FIG5_CBGT_OFFSET = -0.13
_FIG5_EMPIRICAL_OFFSET = 0.13
_FIG5_BOX_WIDTH = 0.18
_FIG5_CAP_WIDTH = _FIG5_BOX_WIDTH * 0.42
_FIG5_RIGHT_AXIS_COLOR = "#0176BEFF"
_FIG5_SIGNIFICANCE_SHOW_NS = False
_FIG5_SIGNIFICANCE_FONT_SIZE = 9.5
_FIG5_SIGNIFICANCE_LINEWIDTH = 1.0

_FIG5_PAIRWISE_COMPARISONS = {
    "A": (("i_only", "left"), ("i_only", "right")),
    "B": (("8", "8_to_10"), ("4", "4_to_5")),
    "C": (("state10", "10_to_11"), ("state5", "5_to_7")),
}


def _fig5_as_float(value):
    try:
        value = float(value)
    except Exception:
        return np.nan
    return value if np.isfinite(value) else np.nan


def _fig5_prediction_rows(results: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if results is None or results.empty:
        return pd.DataFrame()
    return results.loc[results["prediction"].astype(str).str.startswith(prefix)].copy()


def _fig5_prepare_panel_a(results: pd.DataFrame) -> pd.DataFrame:
    panel = _fig5_prediction_rows(results, "1")
    if panel.empty:
        return panel
    panel["plot_key"] = panel["group"].map(
        {
            "GU_visited_1_2_3": "i_only",
            "GL_visited_8_or_10": "left",
            "GR_visited_4_or_5": "right",
        }
    ).fillna(panel["group"].astype(str))
    return panel


def _fig5_prepare_panel_b(results: pd.DataFrame) -> pd.DataFrame:
    panel = _fig5_prediction_rows(results, "2")
    if panel.empty:
        return panel

    def key_from_metric(metric: object) -> str:
        text = str(metric)
        if "previous_state=8" in text and "current_state=10" in text:
            return "8_to_10"
        if "previous_state=4" in text and "current_state=5" in text:
            return "4_to_5"
        if "current_state=8" in text:
            return "8"
        if "current_state=4" in text:
            return "4"
        return text

    panel["plot_key"] = panel["metric"].map(key_from_metric)
    return panel


def _fig5_rows_in_order(frame: pd.DataFrame, keys: list[str]) -> list[pd.Series | None]:
    rows: list[pd.Series | None] = []
    for key in keys:
        if frame is None or frame.empty or "plot_key" not in frame.columns:
            rows.append(None)
            continue
        selected = frame.loc[frame["plot_key"].eq(key)]
        rows.append(selected.iloc[0] if len(selected) else None)
    return rows


def _fig5_q_from_summary_row(row: pd.Series | None) -> tuple[float, float, float, float, float]:
    if row is None:
        return (np.nan,) * 5
    return (
        _fig5_as_float(row.get("q02_5", row.get("ci95_lower", np.nan))),
        _fig5_as_float(row.get("q25", np.nan)),
        _fig5_as_float(row.get("q50", row.get("observed", np.nan))),
        _fig5_as_float(row.get("q75", np.nan)),
        _fig5_as_float(row.get("q97_5", row.get("ci95_upper", np.nan))),
    )


def _fig5_summary_q_lookup(frame: pd.DataFrame, keys: list[str]) -> dict[str, tuple[float, ...]]:
    return {
        key: _fig5_q_from_summary_row(row)
        for key, row in zip(keys, _fig5_rows_in_order(frame, keys))
    }


def _fig5_p23_arrays(raw: pd.DataFrame, value_col: str = "reaction_time") -> list[dict[str, object]]:
    order = (
        ("left_channel_10_then_later_11", "A", "state10", "Left (d+i)"),
        ("left_channel_10_then_later_11", "B", "10_to_11", "+ Right i"),
        ("right_channel_5_then_later_7", "A", "state5", "Right (d+i)"),
        ("right_channel_5_then_later_7", "B", "5_to_7", "+ Left i"),
    )
    rows: list[dict[str, object]] = []
    for comparison, group, key, label in order:
        if raw is None or raw.empty or value_col not in raw.columns:
            values = np.array([], dtype=float)
        else:
            values = (
                pd.to_numeric(
                    raw.loc[
                        raw["comparison"].eq(comparison) & raw["group"].eq(group),
                        value_col,
                    ],
                    errors="coerce",
                )
                .dropna()
                .to_numpy(float)
            )
        rows.append(
            {
                "comparison": comparison,
                "group": group,
                "plot_key": key,
                "label": label,
                "values": values,
            }
        )
    return rows


def _fig5_raw_q_lookup(raw: pd.DataFrame, value_col: str = "reaction_time") -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for row in _fig5_p23_arrays(raw, value_col):
        values = np.asarray(row["values"], dtype=float)
        values = values[np.isfinite(values)]
        quantiles = (
            tuple(np.percentile(values, [2.5, 25, 50, 75, 97.5]))
            if len(values)
            else (np.nan,) * 5
        )
        output[str(row["plot_key"])] = {
            "q": quantiles,
            "label": str(row["label"]),
            "n": int(len(values)),
        }
    return output


def _fig5_draw_percentile_box(
    ax,
    position: float,
    q02_5: float,
    q25: float,
    q50: float,
    q75: float,
    q97_5: float,
    facecolor: str,
    zorder: int = 3,
) -> bool:
    q02_5, q25, q50, q75, q97_5 = [
        _fig5_as_float(value) for value in (q02_5, q25, q50, q75, q97_5)
    ]
    if not np.isfinite(q02_5) or not np.isfinite(q50) or not np.isfinite(q97_5):
        return False
    if not np.isfinite(q25):
        q25 = q50
    if not np.isfinite(q75):
        q75 = q50
    if q75 < q25:
        q25, q75 = q75, q25

    ax.vlines(position, q02_5, q97_5, color=_FIG5_BOX_EDGE, linewidth=1.2, zorder=zorder)
    ax.hlines(
        q02_5,
        position - _FIG5_CAP_WIDTH,
        position + _FIG5_CAP_WIDTH,
        color=_FIG5_BOX_EDGE,
        linewidth=1.2,
        zorder=zorder,
    )
    ax.hlines(
        q97_5,
        position - _FIG5_CAP_WIDTH,
        position + _FIG5_CAP_WIDTH,
        color=_FIG5_BOX_EDGE,
        linewidth=1.2,
        zorder=zorder,
    )
    rectangle = plt.Rectangle(
        (position - _FIG5_BOX_WIDTH / 2, q25),
        _FIG5_BOX_WIDTH,
        max(q75 - q25, 1e-12),
        facecolor=facecolor,
        edgecolor=_FIG5_BOX_EDGE,
        linewidth=1.1,
        alpha=_FIG5_BOX_ALPHA,
        zorder=zorder + 1,
    )
    ax.add_patch(rectangle)
    ax.hlines(
        q50,
        position - _FIG5_BOX_WIDTH / 2,
        position + _FIG5_BOX_WIDTH / 2,
        color="0.05",
        linewidth=1.8,
        zorder=zorder + 2,
    )
    return True


def _fig5_format_panel(ax, title: str, ylim=None, half_line: bool = False) -> None:
    ax.set_title(title, loc="center", pad=20)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if half_line:
        ax.axhline(0.5, linestyle="--", linewidth=1.0, color="0.55", zorder=0)
    ax.grid(axis="y", linewidth=0.6, alpha=0.25)
    ax.tick_params(axis="x", length=0)


def _fig5_finite_extents(q_lookup: dict[str, object], keys: list[str]) -> tuple[float, float]:
    lows: list[float] = []
    highs: list[float] = []
    for key in keys:
        quantiles = q_lookup.get(key, (np.nan,) * 5)
        if isinstance(quantiles, dict):
            quantiles = quantiles["q"]
        low = _fig5_as_float(quantiles[0])
        high = _fig5_as_float(quantiles[4])
        if np.isfinite(low):
            lows.append(low)
        if np.isfinite(high):
            highs.append(high)
    if not lows or not highs:
        return np.nan, np.nan
    return min(lows), max(highs)


def _fig5_nice_ylim(q_lookup: dict[str, object], keys: list[str], pad_frac: float = 0.08) -> tuple[float, float]:
    low, high = _fig5_finite_extents(q_lookup, keys)
    if not np.isfinite(low) or not np.isfinite(high):
        return (0.0, 1.0)
    span = high - low
    if span <= 0:
        span = max(abs(high), 1.0)
    padding = pad_frac * span
    return low - padding, high + padding


def _fig5_auto_right_ylim(
    left_ylim: tuple[float, float],
    left_q: dict[str, object],
    right_q: dict[str, object],
    keys: list[str],
    pad_frac: float = 0.05,
) -> tuple[float, float]:
    left_medians: list[float] = []
    right_medians: list[float] = []
    for key in keys:
        left_values = left_q.get(key, (np.nan,) * 5)
        right_values = right_q.get(key, (np.nan,) * 5)
        if isinstance(left_values, dict):
            left_values = left_values["q"]
        if isinstance(right_values, dict):
            right_values = right_values["q"]
        left_median = _fig5_as_float(left_values[2])
        right_median = _fig5_as_float(right_values[2])
        if np.isfinite(left_median) and np.isfinite(right_median):
            left_medians.append(left_median)
            right_medians.append(right_median)

    left_medians_array = np.asarray(left_medians, dtype=float)
    right_medians_array = np.asarray(right_medians, dtype=float)
    if len(left_medians_array) >= 2 and np.ptp(right_medians_array) > 1e-12:
        slope, intercept = np.polyfit(right_medians_array, left_medians_array, 1)
        if np.isfinite(slope) and abs(slope) > 1e-12:
            right_low = (left_ylim[0] - intercept) / slope
            right_high = (left_ylim[1] - intercept) / slope
            if right_low > right_high:
                right_low, right_high = right_high, right_low
        else:
            right_low, right_high = np.nan, np.nan
    else:
        right_low, right_high = np.nan, np.nan

    if not np.isfinite(right_low) or not np.isfinite(right_high):
        right_low, right_high = _fig5_finite_extents(right_q, keys)

    quantile_low, quantile_high = _fig5_finite_extents(right_q, keys)
    if np.isfinite(quantile_low) and np.isfinite(quantile_high):
        span = max(quantile_high - quantile_low, 1e-9)
        padding = pad_frac * span
        right_low = min(right_low, quantile_low - padding)
        right_high = max(right_high, quantile_high + padding)
    return float(right_low), float(right_high)


def _fig5_style_axes(ax_left, ax_right, ylabel: str) -> None:
    ax_left.set_ylabel(ylabel, color=_FIG5_BOX_EDGE)
    ax_left.tick_params(axis="y", colors=_FIG5_BOX_EDGE)
    ax_left.spines["left"].set_color(_FIG5_BOX_EDGE)
    ax_left.spines["left"].set_linewidth(1.2)
    ax_right.spines["right"].set_visible(True)
    ax_right.spines["right"].set_color(_FIG5_RIGHT_AXIS_COLOR)
    ax_right.spines["right"].set_linewidth(1.2)
    ax_right.set_ylabel("")
    ax_right.tick_params(axis="y", colors=_FIG5_RIGHT_AXIS_COLOR)
    ax_right.grid(False)


def _fig5_plot_twin_bootstrap_panel(
    ax_left,
    cbgt_frame: pd.DataFrame,
    empirical_frame: pd.DataFrame,
    keys: list[str],
    labels: dict[str, str],
    title: str,
    ylabel: str,
    panel: str,
    left_ylim: tuple[float, float] = (0.0, 1.0),
    half_line: bool = True,
):
    ax_right = ax_left.twinx()
    cbgt_q = _fig5_summary_q_lookup(cbgt_frame, keys)
    empirical_q = _fig5_summary_q_lookup(empirical_frame, keys)
    base_positions = np.arange(len(keys))
    plotted_rows: list[dict[str, object]] = []

    for position, key in zip(base_positions, keys):
        quantiles = cbgt_q[key]
        if _fig5_draw_percentile_box(
            ax_left,
            position + _FIG5_CBGT_OFFSET,
            *quantiles,
            facecolor=_FIG5_CBGT_BOX_FACE,
        ):
            plotted_rows.append(
                {
                    "panel": panel,
                    "source": "CBGT",
                    "plot_key": key,
                    **dict(zip(("q02_5", "q25", "q50", "q75", "q97_5"), quantiles)),
                }
            )
        quantiles = empirical_q[key]
        if _fig5_draw_percentile_box(
            ax_right,
            position + _FIG5_EMPIRICAL_OFFSET,
            *quantiles,
            facecolor=_FIG5_EMPIRICAL_BOX_FACE,
        ):
            plotted_rows.append(
                {
                    "panel": panel,
                    "source": "IBL+Steinmetz",
                    "plot_key": key,
                    **dict(zip(("q02_5", "q25", "q50", "q75", "q97_5"), quantiles)),
                }
            )

    ax_left.set_xticks(base_positions)
    ax_left.set_xticklabels([labels.get(key, key) for key in keys])
    ax_left.set_xlim(base_positions[0] - 0.45, base_positions[-1] + 0.45)
    _fig5_format_panel(ax_left, title, ylim=left_ylim, half_line=half_line)
    right_ylim = _fig5_auto_right_ylim(left_ylim, cbgt_q, empirical_q, keys)
    ax_right.set_ylim(*right_ylim)

    if panel == "A":
        ax_right.axhline(
            0.5,
            linestyle="--",
            linewidth=1.0,
            color=_FIG5_RIGHT_AXIS_COLOR,
            alpha=0.9,
            zorder=0,
        )
    if panel == "B":
        ax_right.set_yticks([tick for tick in ax_right.get_yticks() if tick >= 0])

    _fig5_style_axes(ax_left, ax_right, ylabel)
    return ax_right, pd.DataFrame(plotted_rows)


def _fig5_plot_twin_raw_panel(
    ax_left,
    cbgt_raw: pd.DataFrame,
    empirical_raw: pd.DataFrame,
    value_col: str = "reaction_time",
):
    ax_right = ax_left.twinx()
    cbgt_q = _fig5_raw_q_lookup(cbgt_raw, value_col)
    empirical_q = _fig5_raw_q_lookup(empirical_raw, value_col)
    keys = ["state10", "10_to_11", "state5", "5_to_7"]
    base_positions = np.arange(1, len(keys) + 1)
    plotted_rows: list[dict[str, object]] = []

    for position, key in zip(base_positions, keys):
        quantiles = cbgt_q[key]["q"]
        if _fig5_draw_percentile_box(
            ax_left,
            position + _FIG5_CBGT_OFFSET,
            *quantiles,
            facecolor=_FIG5_CBGT_BOX_FACE,
        ):
            plotted_rows.append(
                {
                    "panel": "C",
                    "source": "CBGT",
                    "plot_key": key,
                    "n_trials": cbgt_q[key]["n"],
                    **dict(zip(("q02_5", "q25", "q50", "q75", "q97_5"), quantiles)),
                }
            )
        quantiles = empirical_q[key]["q"]
        if _fig5_draw_percentile_box(
            ax_right,
            position + _FIG5_EMPIRICAL_OFFSET,
            *quantiles,
            facecolor=_FIG5_EMPIRICAL_BOX_FACE,
        ):
            plotted_rows.append(
                {
                    "panel": "C",
                    "source": "IBL+Steinmetz",
                    "plot_key": key,
                    "n_trials": empirical_q[key]["n"],
                    **dict(zip(("q02_5", "q25", "q50", "q75", "q97_5"), quantiles)),
                }
            )

    labels = [
        "Left (d+i)",
        "Left (d+i)\n+ Right i",
        "Right (d+i)",
        "Right (d+i)\n+ Left i",
    ]
    ax_left.set_xticks(base_positions)
    ax_left.set_xticklabels(labels)
    ax_left.set_xlim(0.35, len(keys) + 0.65)
    left_ylim = _fig5_nice_ylim(cbgt_q, keys, pad_frac=0.08)
    _fig5_format_panel(
        ax_left,
        "(C) Slower decisions with opponent iSPN",
        ylim=left_ylim,
        half_line=False,
    )
    ax_left.axvline(2.5, linestyle=":", linewidth=1.5, color="0.65", zorder=0)
    ax_right.set_ylim(97, 406)
    _fig5_style_axes(ax_left, ax_right, "Decision time (ms)")
    return ax_right, pd.DataFrame(plotted_rows)


def _fig5_significance_row(
    pairwise: pd.DataFrame,
    panel: str,
    source: str,
    group1: str,
    group2: str,
) -> pd.Series | None:
    if pairwise is None or pairwise.empty:
        return None
    selected = pairwise.loc[
        pairwise["panel"].eq(panel)
        & pairwise["source"].eq(source)
        & pairwise["group1"].eq(group1)
        & pairwise["group2"].eq(group2)
    ]
    return selected.iloc[0] if len(selected) else None


def _fig5_draw_bracket(ax, x1, x2, y_fraction, label, color, height_fraction=0.025) -> None:
    if label == "n.s." and not _FIG5_SIGNIFICANCE_SHOW_NS:
        return
    if not label:
        return
    transform = ax.get_xaxis_transform()
    ax.plot(
        [x1, x1, x2, x2],
        [y_fraction, y_fraction + height_fraction, y_fraction + height_fraction, y_fraction],
        transform=transform,
        color=color,
        linewidth=_FIG5_SIGNIFICANCE_LINEWIDTH,
        clip_on=False,
        zorder=20,
    )
    label_offset = 0 if label == "n.s." else 0.006
    ax.text(
        (x1 + x2) / 2,
        y_fraction + height_fraction - label_offset,
        label,
        ha="center",
        va="bottom",
        color=color,
        fontsize=_FIG5_SIGNIFICANCE_FONT_SIZE,
        transform=transform,
        clip_on=False,
        zorder=21,
    )


def _fig5_draw_z_bracket(ax, x1, x2, y_fraction, label, color, height_fraction=0.025) -> None:
    if label == "n.s." and not _FIG5_SIGNIFICANCE_SHOW_NS:
        return
    if not label:
        return
    transform = ax.get_xaxis_transform()
    ax.plot(
        [x1, x1, x2, x2],
        [
            y_fraction + height_fraction,
            y_fraction,
            y_fraction,
            y_fraction - height_fraction,
        ],
        transform=transform,
        color=color,
        linewidth=_FIG5_SIGNIFICANCE_LINEWIDTH,
        clip_on=False,
        zorder=20,
    )
    ax.text(
        (x1 + x2) / 2,
        y_fraction + height_fraction - 0.024,
        label,
        ha="center",
        va="bottom",
        color=color,
        fontsize=_FIG5_SIGNIFICANCE_FONT_SIZE,
        transform=transform,
        clip_on=False,
        zorder=21,
    )


def _fig5_annotate_significance(
    ax,
    pairwise: pd.DataFrame,
    panel: str,
    keys: list[str],
    base_start: int,
    sources: tuple[str, ...],
) -> None:
    base_positions = dict(zip(keys, np.arange(len(keys)) + base_start))
    source_offsets = {
        "CBGT": _FIG5_CBGT_OFFSET,
        "IBL+Steinmetz": _FIG5_EMPIRICAL_OFFSET,
    }
    source_colors = {
        "CBGT": _FIG5_BOX_EDGE,
        "IBL+Steinmetz": _FIG5_RIGHT_AXIS_COLOR,
    }
    source_y_shift = {"CBGT": 0.0, "IBL+Steinmetz": 0.040}

    for group1, group2 in _FIG5_PAIRWISE_COMPARISONS[panel]:
        for source in sources:
            row = _fig5_significance_row(pairwise, panel, source, group1, group2)
            if row is None:
                continue
            label = str(row.get("significance", ""))
            x1 = base_positions[group1] + source_offsets[source]
            x2 = base_positions[group2] + source_offsets[source]
            color = source_colors[source]
            if panel == "A" and group2 == "right":
                _fig5_draw_z_bracket(
                    ax,
                    x1,
                    x2,
                    0.35 + source_y_shift[source],
                    label,
                    color,
                    height_fraction=0.018,
                )
            elif panel == "A":
                _fig5_draw_bracket(
                    ax,
                    x1,
                    x2,
                    0.94 + source_y_shift[source],
                    label,
                    color,
                    height_fraction=0.022,
                )
            else:
                _fig5_draw_bracket(
                    ax,
                    x1,
                    x2,
                    0.94 + source_y_shift[source],
                    label,
                    color,
                    height_fraction=0.022,
                )


def plot_prediction_boxplots_exact(
    cbgt_main: pd.DataFrame,
    empirical_main: pd.DataFrame,
    cbgt_raw: pd.DataFrame,
    empirical_raw: pd.DataFrame,
    pairwise_tests: pd.DataFrame,
):

    rc = {
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "font.size": 13,
        "axes.labelsize": 14.4,
        "axes.titlesize": 15.5,
        "xtick.labelsize": 13.4,
        "ytick.labelsize": 12.5,
        "legend.fontsize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
    with mpl.rc_context(rc):
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(16, 6.5),
            constrained_layout=False,
            gridspec_kw={"width_ratios": [2.6, 3.3, 4.1]},
        )
        fig.subplots_adjust(
            left=0.055,
            right=0.975,
            bottom=0.15,
            top=0.80,
            wspace=0.33,
        )
        ax1, ax2, ax3 = axes.flat
        plotted_tables: list[pd.DataFrame] = []

        ax1_right, plotted = _fig5_plot_twin_bootstrap_panel(
            ax1,
            _fig5_prepare_panel_a(cbgt_main),
            _fig5_prepare_panel_a(empirical_main),
            keys=["i_only", "left", "right"],
            labels={
                "i_only": "iSPN-only",
                "left": "Left d\nLeft (d+i)",
                "right": "Right d\nRight (d+i)",
            },
            title="(A) iSPN-only states preserve uncertainty",
            ylabel="Left-choice probability",
            panel="A",
            left_ylim=(0, 1),
            half_line=True,
        )
        plotted_tables.append(plotted)

        ax2_right, plotted = _fig5_plot_twin_bootstrap_panel(
            ax2,
            _fig5_prepare_panel_b(cbgt_main),
            _fig5_prepare_panel_b(empirical_main),
            keys=["8", "8_to_10", "4", "4_to_5"],
            labels={
                "8": "Left d",
                "8_to_10": "Left (d+i)",
                "4": "Right d",
                "4_to_5": "Right (d+i)",
            },
            title="(B) Same-channel d+i marks terminality",
            ylabel="Termination probability",
            panel="B",
            left_ylim=(0, 1),
            half_line=False,
        )
        ax2.axvline(1.5, linestyle=":", linewidth=1.5, color="0.65", zorder=0)
        plotted_tables.append(plotted)

        ax3_right, plotted = _fig5_plot_twin_raw_panel(
            ax3,
            cbgt_raw,
            empirical_raw,
            value_col="reaction_time",
        )
        plotted_tables.append(plotted)

        _fig5_annotate_significance(
            ax1,
            pairwise_tests,
            panel="A",
            keys=["i_only", "left", "right"],
            base_start=0,
            sources=("CBGT",),
        )
        _fig5_annotate_significance(
            ax1_right,
            pairwise_tests,
            panel="A",
            keys=["i_only", "left", "right"],
            base_start=0,
            sources=("IBL+Steinmetz",),
        )
        _fig5_annotate_significance(
            ax2,
            pairwise_tests,
            panel="B",
            keys=["8", "8_to_10", "4", "4_to_5"],
            base_start=0,
            sources=("CBGT",),
        )
        _fig5_annotate_significance(
            ax2_right,
            pairwise_tests,
            panel="B",
            keys=["8", "8_to_10", "4", "4_to_5"],
            base_start=0,
            sources=("IBL+Steinmetz",),
        )
        _fig5_annotate_significance(
            ax3,
            pairwise_tests,
            panel="C",
            keys=["state10", "10_to_11", "state5", "5_to_7"],
            base_start=1,
            sources=("CBGT",),
        )
        _fig5_annotate_significance(
            ax3_right,
            pairwise_tests,
            panel="C",
            keys=["state10", "10_to_11", "state5", "5_to_7"],
            base_start=1,
            sources=("IBL+Steinmetz",),
        )

        handles = [
            mpl.patches.Patch(
                facecolor=_FIG5_CBGT_BOX_FACE,
                edgecolor=_FIG5_BOX_EDGE,
                label="CBGT, left axis",
            ),
            mpl.patches.Patch(
                facecolor=_FIG5_EMPIRICAL_BOX_FACE,
                edgecolor=_FIG5_BOX_EDGE,
                label="IBL+Steinmetz, right axis",
            ),
        ]
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.96),
            ncol=2,
            fontsize=15,
            frameon=False,
        )
        fig.text(
            0.5,
            0.057,
            "Boxplots: box = central 50% of trials or bootstrap samples; "
            "whiskers = central 95%; horizontal line = median.",
            ha="center",
            va="center",
            fontsize=14.4,
        )
        fig.text(
            0.5,
            0.025,
            "Significance: * p < 0.05; ** p < 0.01; *** p < 0.001.",
            ha="center",
            va="center",
            fontsize=14.4,
        )

        for axis in (ax1, ax2, ax3):
            axis.tick_params(axis="x", pad=4)
            for tick in axis.get_xticklabels():
                tick.set_fontweight("normal")
                tick.set_linespacing(0.88)

        plotted_summary = pd.concat(
            [table for table in plotted_tables if table is not None and len(table)],
            ignore_index=True,
        )
    return fig, plotted_summary
