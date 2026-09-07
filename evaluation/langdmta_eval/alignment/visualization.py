"""Visualization functions for human-LLM alignment and inter-annotator agreement."""

import math
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from langdmta_eval.alignment.metrics import AlignmentResult, InterAnnotatorResult

plt.rcParams.update({"font.size": 16})

TITLE = False  # Set to True to display titles

DISCRETE_COLORS = ["#90CAF9", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]


def _clean_metric_name(metric: str) -> str:
    """Convert metric name to display format."""
    return metric.replace("_", " ").title()


def _label_map(value: float) -> str:
    """Convert numeric score to short display label."""
    if value == -1.0:
        return "N/A"
    if value == 0.0:
        return "0.0"
    if value == 0.5:
        return "0.5"
    if value == 1.0:
        return "1.0"
    return str(value)


def plot_confusion_matrix(
    result: AlignmentResult,
    normalize: bool = False,
    figsize: Tuple[int, int] = (6, 5),
    cmap: str = "Blues",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot a single confusion matrix heatmap for one metric.

    Rows = human scores, Columns = LLM scores.

    Args:
        result: AlignmentResult for one metric
        normalize: If True, row-normalize to show proportions
        figsize: Figure size
        cmap: Colormap name
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    if normalize and result.confusion_matrix.sum() > 0:
        cm = result.confusion_matrix.astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid division by zero
        cm = cm / row_sums
        fmt = ".2f"
    else:
        cm = result.confusion_matrix.astype(int)
        fmt = "d"

    tick_labels = [_label_map(v) for v in result.labels]

    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )

    ax.set_xlabel("LLM Judge Score")
    ax.set_ylabel("Human Score")
    if TITLE:
        ax.set_title(
            f"{_clean_metric_name(result.metric_name)}\n"
            f"(n={result.n_samples}, "
            f"\u03ba={result.cohens_kappa_weighted:.2f})"
            if not np.isnan(result.cohens_kappa_weighted)
            else f"{_clean_metric_name(result.metric_name)}\n(n={result.n_samples})"
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_all_confusion_matrices(
    results: Dict[str, AlignmentResult],
    normalize: bool = False,
    figsize: Tuple[int, int] = (18, 12),
    cmap: str = "Blues",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot confusion matrices for all metrics in a grid layout (Figure 4).

    Args:
        results: Dict mapping metric_name -> AlignmentResult
        normalize: If True, row-normalize
        figsize: Figure size
        cmap: Colormap name
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure object
    """
    metrics = list(results.keys())
    n = len(metrics)
    n_cols = min(3, n)
    n_rows = math.ceil(n / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        r = results[metric]

        if normalize and r.confusion_matrix.sum() > 0:
            cm = r.confusion_matrix.astype(float)
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            cm = cm / row_sums
            fmt = ".2f"
        else:
            cm = r.confusion_matrix.astype(int)
            fmt = "d"

        tick_labels = [_label_map(v) for v in r.labels]

        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            xticklabels=tick_labels,
            yticklabels=tick_labels,
            linewidths=0.5,
            linecolor="white",
            ax=ax,
        )

        kappa_str = (
            f"\u03ba={r.cohens_kappa_weighted:.2f}"
            if not np.isnan(r.cohens_kappa_weighted)
            else "\u03ba=N/A"
        )
        # if TITLE:
        ax.set_title(f"{_clean_metric_name(metric)}\n(n={r.n_samples}, {kappa_str})")
        ax.set_xlabel("LLM Judge Score")
        ax.set_ylabel("Human Score")

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Human vs LLM Judge Agreement", fontsize=18, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_alignment_bar_chart(
    results: Dict[str, AlignmentResult],
    metric_key: str = "cohens_kappa_weighted",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart of alignment scores per evaluation metric.

    Args:
        results: Dict mapping metric_name -> AlignmentResult
        metric_key: Which agreement metric to plot. One of:
            'cohens_kappa_weighted', 'cohens_kappa_unweighted',
            'exact_match_rate', 'spearman_correlation'
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure object
    """
    metrics = list(results.keys())
    values = [getattr(results[m], metric_key, np.nan) for m in metrics]
    labels = [_clean_metric_name(m) for m in metrics]

    fig, ax = plt.subplots(figsize=figsize)

    colors = []
    for v in values:
        if np.isnan(v):
            colors.append("#cccccc")
        elif v >= 0.6:
            colors.append(DISCRETE_COLORS[1])
        elif v >= 0.4:
            colors.append(DISCRETE_COLORS[2])
        else:
            colors.append(DISCRETE_COLORS[4])

    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.margins(x=0.05)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        if not np.isnan(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
            )

    display_name = metric_key.replace("_", " ").title()
    ax.set_ylabel(display_name)
    if TITLE:
        ax.set_title(f"Human-LLM Alignment: {display_name}")
    ax.set_ylim(-0.1, 1.1)
    ax.axhline(
        y=0.6, color="green", linestyle="--", alpha=0.5, label="Substantial (0.6)"
    )
    ax.axhline(y=0.4, color="orange", linestyle="--", alpha=0.5, label="Moderate (0.4)")
    ax.legend(fontsize=14)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="center")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_alignment_comparison(
    before_results: Dict[str, AlignmentResult],
    after_results: Dict[str, AlignmentResult],
    labels: Tuple[str, str] = ("Baseline", "Optimized"),
    metric_key: str = "cohens_kappa_weighted",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Grouped bar chart comparing before/after optimization alignment (Figure 5).

    Args:
        before_results: Alignment results before optimization
        after_results: Alignment results after optimization
        labels: Labels for the two conditions
        metric_key: Which agreement metric to plot
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure object
    """
    metrics = list(before_results.keys())
    before_vals = [getattr(before_results[m], metric_key, np.nan) for m in metrics]
    after_vals = [
        getattr(
            after_results.get(
                m,
                AlignmentResult(
                    m, 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.zeros((3, 3))
                ),
            ),
            metric_key,
            np.nan,
        )
        for m in metrics
    ]
    display_labels = [_clean_metric_name(m) for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)

    bars1 = ax.bar(
        x - width / 2,
        before_vals,
        width,
        label=labels[0],
        color=DISCRETE_COLORS[0],
        edgecolor="black",
        linewidth=0.5,
    )
    bars2 = ax.bar(
        x + width / 2,
        after_vals,
        width,
        label=labels[1],
        color=DISCRETE_COLORS[1],
        edgecolor="black",
        linewidth=0.5,
    )

    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.02,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    display_name = metric_key.replace("_", " ").title()
    ax.set_ylabel(display_name)
    if TITLE:
        ax.set_title(f"Alignment Improvement: {labels[0]} vs {labels[1]}")
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, rotation=20, ha="center")
    ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
    ax.set_ylim(-0.1, 1.1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_self_consistency_barplot(
    data: "pd.DataFrame",
    metric: "str | List[str]",
    figsize: Tuple[int, int] = (10, 6),
    color: "str | List[str]" = "skyblue",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar plot of self-consistency scores grouped by evaluation metric.

    Args:
        data: DataFrame with columns 'metric' and the specified metric column(s).
        metric: Column name or list of column names to plot on the y-axis.
        figsize: Figure size.
        color: Bar fill color or list of colors (one per metric).
        save_path: Optional path to save figure.

    Returns:
        matplotlib Figure object.
    """
    metrics = [metric] if isinstance(metric, str) else list(metric)
    if isinstance(color, str):
        colors = DISCRETE_COLORS[: len(metrics)] if len(metrics) > 1 else [color]
    else:
        colors = list(color)

    fig, ax = plt.subplots(figsize=figsize)

    categories = data["metric"].unique()
    x = np.arange(len(categories))
    n_metrics = len(metrics)
    width = 0.9 / n_metrics

    for i, m in enumerate(metrics):
        offset = (i - n_metrics / 2 + 0.5) * width
        vals = [data[data["metric"] == cat][m].mean() for cat in categories]
        ci = []
        for cat in categories:
            group = data[data["metric"] == cat][m].dropna()
            if len(group) > 1:
                se = group.std() / np.sqrt(len(group))
                ci.append(1.96 * se)
            else:
                ci.append(0)
        bars = ax.bar(
            x + offset,
            vals,
            width,
            yerr=ci,
            capsize=4,
            label=_clean_metric_name(m),
            color=colors[i % len(colors)],
            edgecolor="black",
            linewidth=0.5,
        )
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    0,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=14,
                    fontweight="bold",
                )

    display_names = [_clean_metric_name(m) for m in metrics]
    title_label = " & ".join(display_names)
    if TITLE:
        ax.set_title(f"Self-Consistency: {title_label}")
    # ax.set_xlabel("Metric")
    ax.set_ylabel("Agreement Score")  # (" / ".join(display_names))
    ax.set_xticks(x)
    ax.set_xticklabels(
        [_clean_metric_name(cat) for cat in categories], rotation=20, ha="center"
    )
    ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    if n_metrics > 1:
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 0.97),
            ncol=len(metrics),
            frameon=False,
        )
        if TITLE:
            ax.set_title(f"Self-Consistency: {title_label}", pad=30)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    return fig


def plot_inter_annotator_barplot(
    results: Dict[str, "InterAnnotatorResult"],
    metrics: List[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar plot of inter-annotator agreement scores with metrics shown side by side.

    Args:
        results: Dict mapping metric_name -> InterAnnotatorResult.
        metrics: Which agreement measures to plot (default: fleiss_kappa and
            mean_pairwise_kappa).
        figsize: Figure size.
        save_path: Optional path to save figure.

    Returns:
        matplotlib Figure object.
    """
    if metrics is None:
        metrics = ["fleiss_kappa", "mean_pairwise_kappa"]

    categories = list(results.keys())
    x = np.arange(len(categories))
    n_metrics = len(metrics)
    width = 0.9 / n_metrics

    colors = DISCRETE_COLORS[:n_metrics]

    fig, ax = plt.subplots(figsize=figsize)

    for i, m in enumerate(metrics):
        offset = (i - n_metrics / 2 + 0.5) * width
        vals = [getattr(results[cat], m, np.nan) for cat in categories]
        ci = []
        for cat in categories:
            r = results[cat]
            if m == "mean_pairwise_kappa" and r.pairwise_kappas:
                pw_vals = np.array(
                    [v for v in r.pairwise_kappas.values() if not np.isnan(v)]
                )
                if len(pw_vals) > 1:
                    se = pw_vals.std() / np.sqrt(len(pw_vals))
                    ci.append(1.96 * se)
                else:
                    ci.append(0)
            else:
                ci.append(0)
        bars = ax.bar(
            x + offset,
            vals,
            width,
            yerr=ci,
            capsize=4,
            label=_clean_metric_name(m),
            color=colors[i % len(colors)],
            edgecolor="black",
            linewidth=0.5,
        )
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    0,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=14,
                    fontweight="bold",
                )

    display_names = [_clean_metric_name(m) for m in metrics]
    title_label = " & ".join(display_names)
    # ax.set_xlabel("Metric")
    ax.set_ylabel("Agreement Score")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [_clean_metric_name(cat) for cat in categories], rotation=20, ha="center"
    )
    ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=len(metrics),
        frameon=False,
    )
    if TITLE:
        ax.set_title(f"Inter-Annotator Agreement: {title_label}", pad=30)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_llm_inter_judge_barplot(
    results_by_model: Dict[str, Dict[str, "InterAnnotatorResult"]],
    agreement_metric: str = "mean_pairwise_kappa",
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[str] = None,
    error_bars: bool = True,
) -> plt.Figure:
    """Bar plot of LLM inter-judge agreement with models as hue.

    Produces a single figure for one agreement measure.  The x-axis shows
    evaluation metrics (relevancy, completeness, …) and each model is a
    differently-coloured bar group.  Call twice with different
    ``agreement_metric`` values to get two separate plots.

    Args:
        results_by_model: Output of
            :func:`~langdmta_eval.alignment.metrics.compute_all_llm_inter_judge_agreement`.
            Mapping of model_id -> {metric_name -> InterAnnotatorResult}.
        agreement_metric: Which agreement measure to plot. One of
            ``"fleiss_kappa"`` or ``"mean_pairwise_kappa"``.
        figsize: Figure size (width, height).
        save_path: Optional path to save the figure.
        error_bars: Whether to display error bars (95% CI). Default True.

    Returns:
        matplotlib Figure object.
    """
    model_ids = list(results_by_model.keys())
    eval_metrics = list(next(iter(results_by_model.values())).keys())

    n_models = len(model_ids)
    x = np.arange(len(eval_metrics))
    width = 0.9 / n_models

    fig, ax = plt.subplots(figsize=figsize)

    for i, model_id in enumerate(model_ids):
        model_results = results_by_model[model_id]
        vals = []
        ci = []
        for em in eval_metrics:
            r = model_results.get(em)
            if r is None:
                vals.append(np.nan)
                ci.append(0)
                continue
            vals.append(getattr(r, agreement_metric, np.nan))
            if (
                error_bars
                and agreement_metric == "mean_pairwise_kappa"
                and r.pairwise_kappas
            ):
                pw = np.array(
                    [v for v in r.pairwise_kappas.values() if not np.isnan(v)]
                )
                ci.append(1.96 * pw.std() / np.sqrt(len(pw)) if len(pw) > 1 else 0)
            else:
                ci.append(0)

        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            vals,
            width,
            yerr=ci if error_bars else None,
            capsize=4 if error_bars else 0,
            label=model_id,
            color=DISCRETE_COLORS[i % len(DISCRETE_COLORS)],
            edgecolor="black",
            linewidth=0.5,
        )
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    0,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    fontweight="bold",
                )

    ax.set_ylabel(_clean_metric_name(agreement_metric))
    # ax.set_xlabel("Evaluation Metric")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [_clean_metric_name(em) for em in eval_metrics], rotation=20, ha="center"
    )
    ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=n_models,
        frameon=False,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_multi_alignment_comparison(
    results_dict: Dict[str, Dict[str, AlignmentResult]],
    metric_key: str = "cohens_kappa_weighted",
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Grouped bar chart comparing multiple optimization versions (Table 6 visual).

    Args:
        results_dict: Dict mapping version_label -> {metric -> AlignmentResult}
        metric_key: Which agreement metric to plot
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure object
    """
    version_labels = list(results_dict.keys())
    # Use the metrics from the first version as reference
    metrics = list(next(iter(results_dict.values())).keys())
    display_labels = [_clean_metric_name(m) for m in metrics]

    x = np.arange(len(metrics))
    n_versions = len(version_labels)
    width = 0.9 / n_versions

    fig, ax = plt.subplots(figsize=figsize)

    for i, version in enumerate(version_labels):
        vals = []
        for metric in metrics:
            r = results_dict[version].get(metric)
            vals.append(getattr(r, metric_key, np.nan) if r else np.nan)

        offset = (i - n_versions / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            vals,
            width,
            label=version,
            color=DISCRETE_COLORS[i % len(DISCRETE_COLORS)],
            edgecolor="black",
            linewidth=0.5,
        )

        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.02,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=14,
                )

    display_name = metric_key.replace("_", " ").title()
    ax.set_ylabel(display_name)
    if TITLE:
        ax.set_title(f"Alignment Comparison Across Versions: {display_name}")
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, rotation=20, ha="center")
    ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
    ax.set_ylim(-0.1, 1.1)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=len(version_labels),
        frameon=False,
    )
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
