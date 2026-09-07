"""Visualization functions for evaluation results."""

from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from langdmta_eval.schemas import EvaluationResult

plt.rcParams.update({"font.size": 18})

DISCRETE_COLORS = [
    "#90CAF9",
    "#4CAF50",
    "#FF9800",
    "#9C27B0",
    "#F44336",
    "#FFEB3B",
    "#00BCD4",
    "#E91E63",
    "#8BC34A",
    "#795548",
    "#607D8B",
    "#BDBDBD",
]  # Extended color palette
TITLE = False  # Set to True to display titles
# ===== HELPER FUNCTIONS =====


def results_to_dataframe(
    results: Union[EvaluationResult, List[EvaluationResult]],
) -> pd.DataFrame:
    """Convert evaluation results to pandas DataFrame.

    Args:
        results: Single result or list of results

    Returns:
        DataFrame with columns for test case attributes and scores

    Raises:
        ValueError: If results is empty or invalid

    Examples:
        >>> df = results_to_dataframe(evaluation_results)
        >>> df.columns
        Index(['category', 'test_name', 'session', 'question_variation',
               'correctness', 'output_relevancy', ...])
    """
    # Convert single result to list
    if isinstance(results, EvaluationResult):
        results = [results]

    if not results:
        raise ValueError("No results provided")

    # Extract data from each result
    data = []
    for result in results:
        row: Dict[str, Any] = {
            "category": result.test_case.category,
            "test_name": result.test_case.test_name,
            "session": result.test_case.session,
            "question_variation": result.test_case.question_variation,
        }

        # Add all scores
        row.update(result.scores)
        data.append(row)
    return pd.DataFrame(data)


def _get_available_metrics(df: pd.DataFrame) -> List[str]:
    """Extract available metric columns from DataFrame.

    Returns columns that:
    - Have numeric dtype (float or int)
    - Are not in the exclude list (test case attributes)

    Args:
        df: DataFrame with test case attributes and scores

    Returns:
        List of metric column names
    """
    exclude_cols = ["category", "test_name", "session", "question_variation"]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [col for col in numeric_cols if col not in exclude_cols]


def _exclude_not_applicable(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    """Replace -1.0 (NOT APPLICABLE) values with NaN so they are excluded from aggregations.

    Args:
        df: DataFrame with metric columns
        metrics: List of metric column names to filter

    Returns:
        DataFrame with -1.0 values replaced by NaN in metric columns
    """
    df = df.copy()
    for metric in metrics:
        if metric in df.columns:
            df.loc[df[metric] == -1.0, metric] = np.nan
    return df


def _clean_metric_name(metric: str) -> str:
    """Convert metric name to display format.

    Args:
        metric: Raw metric name (e.g., 'output_relevancy')

    Returns:
        Formatted metric name (e.g., 'Output Relevancy')

    Examples:
        >>> _clean_metric_name('output_relevancy')
        'Output Relevancy'
        >>> _clean_metric_name('correctness')
        'Correctness'
    """
    return metric.replace("_", " ").title()


# ===== VISUALIZATION FUNCTIONS =====


def plot_score_histograms(
    results: Union[EvaluationResult, List[EvaluationResult]],
    metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None,
    bins: int = 20,
) -> plt.Figure:
    """Plot histograms of score distributions for each metric.

    Args:
        results: Single result or list of results to visualize
        metrics: List of metrics to plot. If None, plots all available metrics
        figsize: Figure size (width, height)
        save_path: Optional path to save figure
        bins: Number of bins for histogram

    Returns:
        matplotlib Figure object

    Raises:
        ValueError: If no valid scores found

    Examples:
        >>> fig = plot_score_histograms(results)
        >>> fig = plot_score_histograms(results, metrics=['correctness', 'output_relevancy'])
        >>> fig = plot_score_histograms(results, save_path='histograms.png')
    """
    # Convert to DataFrame
    df = results_to_dataframe(results)

    # Get available metrics
    if metrics is None:
        metrics = _get_available_metrics(df)

    if not metrics:
        raise ValueError("No valid scores found")

    # Exclude NOT APPLICABLE (-1.0) values from distributions
    df = _exclude_not_applicable(df, metrics)

    # Determine subplot layout
    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_metrics > 1 else [axes]

    # Plot histogram for each metric
    other_metrics = [m for m in metrics if m != "overall"]
    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        # Filter out NaN values
        if metric == "overall":
            scores = pd.Series(df[other_metrics].values.flatten()).dropna()
        else:
            scores = df[metric].dropna()

        if len(scores) == 0:
            ax.text(
                0.5,
                0.5,
                "No valid scores",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        ax.set_title(f"{_clean_metric_name(metric)}", fontsize=14)

        # Create bar chart with one bar per score category (0, 0.5, 1)
        score_categories = [0, 0.5, 1]
        counts = [int((scores == cat).sum()) for cat in score_categories]
        bar_width = 0.25
        ax.bar(
            score_categories,
            counts,
            width=bar_width,
            edgecolor="black",
            color=DISCRETE_COLORS[idx % len(DISCRETE_COLORS)],
        )
        ax.set_xticks(score_categories)
        ax.set_xticklabels(["0", "0.5", "1"])
        if TITLE:
            ax.set_title(f"Dist. {_clean_metric_name(metric)} Scores")
        ax.set_xlabel("Score")
        ax.set_ylabel("Frequency")
        ax.grid(axis="y", alpha=0.75)

    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    return fig


def plot_score_pie(
    results: Union[EvaluationResult, List[EvaluationResult]],
    metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None,
    bins: int = 20,
) -> plt.Figure:
    """Plot pie charts of score distributions for each metric.

    Args:
        results: Single result or list of results to visualize
        metrics: List of metrics to plot. If None, plots all available metrics
        figsize: Figure size (width, height)
        save_path: Optional path to save figure
        bins: Number of bins for histogram

    Returns:
        matplotlib Figure object

    Raises:
        ValueError: If no valid scores found

    Examples:
        >>> fig = plot_score_histograms(results)
        >>> fig = plot_score_histograms(results, metrics=['correctness', 'output_relevancy'])
        >>> fig = plot_score_histograms(results, save_path='histograms.png')
    """
    # Convert to DataFrame
    df = results_to_dataframe(results)

    # Get available metrics
    if metrics is None:
        metrics = _get_available_metrics(df)

    if not metrics:
        raise ValueError("No valid scores found")

    # Exclude NOT APPLICABLE (-1.0) values from distributions
    df = _exclude_not_applicable(df, metrics)

    # Determine subplot layout
    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_metrics > 1 else [axes]

    # Collect all unique score values across all metrics for consistent color mapping
    other_metrics = [m for m in metrics if m != "overall"]
    all_score_values = set()
    for metric in metrics:
        if metric == "overall":
            vals = pd.Series(df[other_metrics].values.flatten()).dropna()
        else:
            vals = df[metric].dropna()
        all_score_values.update(vals.unique())
    all_score_values = sorted(all_score_values)
    color_map = {
        0: DISCRETE_COLORS[2],
        0.5: DISCRETE_COLORS[0],
        1.0: DISCRETE_COLORS[1],
    }  # Example mapping for common scores #{val: DISCRETE_COLORS[i % len(DISCRETE_COLORS)] for i, val in enumerate(all_score_values)}

    # Plot pie chart for each metric
    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        # Filter out NaN values
        if metric == "overall":
            scores = pd.Series(df[other_metrics].values.flatten()).dropna()
        else:
            scores = df[metric].dropna()

        if len(scores) == 0:
            ax.text(
                0.5,
                0.5,
                "No valid scores",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"Dist. {_clean_metric_name(metric)}")
            continue

        # Create pie chart
        scores = scores.value_counts().sort_index()
        pie_colors = [color_map[val] for val in scores.index]
        wedges, _, _ = ax.pie(
            scores.values,
            labels=scores.index,
            autopct="%1.1f%%",
            colors=pie_colors,
            textprops={"fontsize": 12},
            wedgeprops={"edgecolor": "black", "linewidth": 0.5},
        )
        ax.set_title(f"{_clean_metric_name(metric)}", fontsize=14)

    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    if TITLE:
        plt.suptitle("Score Distribution Across Metrics", fontsize=20)

    # Legend with consistent colors, sorted
    legend_handles = [
        plt.matplotlib.patches.Patch(
            facecolor=color_map[val], edgecolor="black", linewidth=0.5, label=val
        )
        for val in all_score_values
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=len(all_score_values),
        frameon=False,
        fontsize=14,
    )

    plt.subplots_adjust(hspace=0)
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    if save_path:
        plt.savefig(save_path)

    return fig


def plot_score_by_category(
    results: Union[EvaluationResult, List[EvaluationResult]],
    metrics: Optional[List[str]] = None,
    group_by: Literal[
        "category", "test_name", "session", "question_variation"
    ] = "category",
    plot_type: Literal["bar", "box"] = "bar",
    figsize: Tuple[int, int] = (16, 6),
    save_path: Optional[str] = None,
    show_mean: bool = True,
) -> plt.Figure:
    """Plot scores grouped by category, test_name, session, or question_variation.
    Args:
        results: Single result or list of results to visualize
        metrics: List of metrics to plot. If None, plots all available metrics
        group_by: Column to group by ('category', 'test_name', 'session')
        figsize: Figure size (width, height)
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure object

    Raises:
        ValueError: If no valid scores found or invalid group_by value

    Examples:
        >>> fig = plot_score_by_category(results)
        >>> fig = plot_score_by_category(results, group_by='test_name')
        >>> fig = plot_score_by_category(results, metrics=['correctness'])
    """
    # Validate group_by
    valid_groupby = ["category", "test_name", "session", "question_variation"]
    if group_by not in valid_groupby:
        raise ValueError(
            f"Invalid group_by: '{group_by}'. Must be one of: {valid_groupby}"
        )

    # Convert to DataFrame
    df = results_to_dataframe(results)

    print(df.columns)

    print(df.head())

    # Get available metrics
    if metrics is None:
        metrics = _get_available_metrics(df)

    if not metrics:
        raise ValueError("No valid scores found")

    # Exclude NOT APPLICABLE (-1.0) values from aggregations
    df = _exclude_not_applicable(df, metrics)

    # Compute grouped means
    has_overall = "overall" in metrics
    real_metrics = [m for m in metrics if m != "overall"]

    groups = sorted(df[group_by].unique())
    grouped = df.groupby(group_by)[real_metrics].mean()

    if has_overall:
        grouped["overall"] = grouped[real_metrics].mean(axis=1)

    all_metrics = real_metrics + (["overall"] if has_overall else [])

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create grouped bar plot using matplotlib
    x = np.arange(len(all_metrics))
    n_groups = len(groups)
    width = 0.8 / n_groups

    grouped_std = df.groupby(group_by)[real_metrics].std()
    if has_overall:
        grouped_std["overall"] = grouped_std[real_metrics].mean(axis=1)

    for i, group in enumerate(groups):
        offset = (i - n_groups / 2 + 0.5) * width
        vals = [float(grouped.loc[group, m]) for m in all_metrics]
        stds = [
            (
                float(grouped_std.loc[group, m])
                if not np.isnan(grouped_std.loc[group, m])
                else 0
            )
            for m in all_metrics
        ]
        bars = ax.bar(
            x + offset,
            vals,
            width,
            yerr=stds,
            capsize=3,
            label=str(group),
            color=DISCRETE_COLORS[i % len(DISCRETE_COLORS)],
            edgecolor="black",
            linewidth=0.5,
        )
        if show_mean:
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        0.01,
                        f"{val:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                        fontweight="bold",
                        zorder=10,
                    )

    ax.set_xticks(x)
    ax.set_xticklabels([_clean_metric_name(m) for m in all_metrics])

    # Styling
    if TITLE:
        ax.set_title(
            f"Score Distribution by {group_by.replace('_', ' ').title()}", pad=30
        )
    # ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_ylim((-0.1, 1.1))
    ax.grid(axis="y", alpha=0.75)
    if n_groups <= 5:
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 0.97),
            ncol=n_groups,
            frameon=False,
        )
    else:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(rotation=20, ha="center")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    return fig


def plot_lineplot_variation(
    results: Union[EvaluationResult, List[EvaluationResult]],
    group_by: Literal[
        "category", "test_name", "session", "question_variation"
    ] = "question_variation",
    metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (16, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot scores grouped by category, test_name, session, or question_variation.
    Args:
        results: Single result or list of results to visualize
        metrics: List of metrics to plot. If None, plots all available metrics
        figsize: Figure size (width, height)
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure object

    Raises:
        ValueError: If no valid scores found or invalid group_by value

    Examples:
        >>> fig = plot_score_by_category(results)
        >>> fig = plot_score_by_category(results, group_by='test_name')
        >>> fig = plot_score_by_category(results, metrics=['correctness'])
    """
    # Validate group_by
    # group_by = "question_variation"

    # Convert to DataFrame
    df = results_to_dataframe(results)

    # Get available metrics
    if metrics is None:
        metrics = _get_available_metrics(df)

    if not metrics:
        raise ValueError("No valid scores found")

    # Exclude NOT APPLICABLE (-1.0) values from aggregations
    df = _exclude_not_applicable(df, metrics)

    # Melt DataFrame to long format
    has_overall = "overall" in metrics
    real_metrics = [m for m in metrics if m != "overall"]

    df_long = df.melt(
        id_vars=[group_by],
        value_vars=real_metrics,
        var_name="metric",
        value_name="score",
    )

    if group_by == "question_variation":
        df_long.rename(columns={"question_variation": "formality_level"}, inplace=True)
        group_by = "formality_level"

    if has_overall:
        overall_rows = df_long.copy()
        overall_rows["metric"] = "overall"
        df_long = pd.concat([df_long, overall_rows], ignore_index=True)

    # Clean up metric names
    df_long["metric"] = df_long["metric"].apply(_clean_metric_name)

    # Filter out NaN values
    df_long = df_long.dropna(subset=["score"])

    if len(df_long) == 0:
        raise ValueError("No valid scores found after filtering")

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    sns.lineplot(
        x=group_by,
        y="score",
        hue="metric",
        data=df_long,
        marker="o",
        ax=ax,
        palette=DISCRETE_COLORS,
    )

    # Styling
    ax.set_title(f"Score Distribution by {group_by.replace('_', ' ').title()}")
    ax.set_xlabel(group_by.replace("_", " ").title())
    ax.set_ylabel("Score")
    ax.set_ylim((-0.1, 1.1))
    ax.grid(axis="y", alpha=0.75)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(rotation=20, ha="center")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    return fig


def plot_score_heatmap(
    results: Union[EvaluationResult, List[EvaluationResult]],
    metric: str,
    group_by: Literal[
        "category", "test_name", "session", "question_variation"
    ] = "category",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    show_all_row: bool = True,
    pool_metrics: Optional[List[str]] = None,
) -> plt.Figure:
    """Plot heatmap of score value frequencies by category.

    Args:
        results: Single result or list of results to visualize
        metric: Metric to visualize (e.g., 'correctness', 'output_relevancy')
        group_by: Column to group by ('category', 'test_name', 'session', 'question_variation')
        figsize: Figure size (width, height)
        save_path: Optional path to save figure
        show_all_row: Whether to add an "All" row showing totals

    Returns:
        matplotlib Figure object

    Raises:
        ValueError: If metric not found or no valid data

    Examples:
        >>> fig = plot_score_heatmap(results, metric='correctness')
        >>> fig = plot_score_heatmap(results, metric='ethical_awareness',
        ...                           group_by='test_name')
    """
    # Validate group_by
    valid_groupby = ["category", "test_name", "session", "question_variation"]
    if group_by not in valid_groupby:
        raise ValueError(
            f"Invalid group_by: '{group_by}'. Must be one of: {valid_groupby}"
        )

    # Convert to DataFrame
    df = results_to_dataframe(results)

    # Handle "overall" by pooling all specified metrics
    if metric == "overall":
        pool_cols = pool_metrics if pool_metrics else _get_available_metrics(df)
        df = _exclude_not_applicable(df, pool_cols)
        df_filtered = df.melt(
            id_vars=[group_by],
            value_vars=pool_cols,
            var_name="_metric",
            value_name="overall",
        )[[group_by, "overall"]].dropna()
    else:
        # Verify metric exists
        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found in results")
        # Filter out NaN values
        df_filtered = df[[group_by, metric]].dropna()

    if len(df_filtered) == 0:
        raise ValueError(f"No valid scores found for metric '{metric}'")

    # Define score values and labels (include NOT APPLICABLE if present)
    has_na = (df_filtered[metric] == -1.0).any()
    if has_na:
        score_values = [-1.0, 0.0, 0.5, 1.0]
        score_labels = ["-1.0 (N/A)", "0.0 (Low)", "0.5 (Medium)", "1.0 (High)"]
    else:
        score_values = [0.0, 0.5, 1.0]
        score_labels = ["0.0 (Low)", "0.5 (Medium)", "1.0 (High)"]

    # Get unique groups
    groups = sorted(df_filtered[group_by].unique())

    # Build frequency matrix
    matrix_data = []
    for group in groups:
        group_scores = df_filtered[df_filtered[group_by] == group][metric]
        row = [(group_scores == score_val).sum() for score_val in score_values]
        matrix_data.append(row)

    # Add "All" row if requested
    if show_all_row:
        all_scores = df_filtered[metric]
        all_row = [(all_scores == score_val).sum() for score_val in score_values]
        matrix_data.append(all_row)
        groups = list(groups) + ["All"]

    # Create DataFrame for heatmap
    heatmap_df = pd.DataFrame(matrix_data, index=groups, columns=score_labels)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt="d",
        cmap="Greens",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Frequency"},
        ax=ax,
    )

    # Styling
    ax.set_title(f"Score Distribution for {_clean_metric_name(metric)}")
    ax.set_xlabel("Score Value")
    ax.set_ylabel(group_by.replace("_", " ").title())

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    return fig


def plot_metric_correlation(
    results: Union[EvaluationResult, List[EvaluationResult]],
    metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot correlation matrix between evaluation metrics.

    Shows how different evaluation metrics correlate with each other using a heatmap.
    This helps identify which metrics tend to move together and which are independent.

    Correlation values range from -1 (perfect negative correlation) to +1 (perfect positive
    correlation). Cells showing "N/A" indicate undefined correlations, which occur when:
    - A metric has zero variance (all same values)
    - Insufficient overlapping non-NaN values between two metrics

    Args:
        results: Single result or list of results to visualize
        metrics: List of metrics to include. If None, uses all available metrics
        figsize: Figure size (width, height)
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure object

    Raises:
        ValueError: If insufficient data or no valid metrics found

    Examples:
        >>> fig = plot_metric_correlation(results)
        >>> fig = plot_metric_correlation(results, metrics=['correctness', 'output_relevancy'])
        >>> fig = plot_metric_correlation(results, save_path='correlation.png')
    """
    # Convert to DataFrame
    df = results_to_dataframe(results)

    # Get available metrics
    if metrics is None:
        metrics = _get_available_metrics(df)

    # "overall" is a derived metric — exclude from correlation
    metrics = [m for m in metrics if m != "overall"]

    if not metrics:
        raise ValueError("No valid metrics found")

    if len(metrics) < 2:
        raise ValueError("Need at least 2 metrics for correlation analysis")

    # Exclude NOT APPLICABLE (-1.0) values from correlations
    df = _exclude_not_applicable(df, metrics)

    # Filter to selected metrics
    df_metrics = df[metrics]

    # Calculate correlation matrix using pairwise complete observations
    # This handles missing values better by computing correlation for each pair independently
    corr = df_metrics.corr(method="pearson", min_periods=2)

    # Check if we have any valid correlations
    if corr.isna().all().all():
        raise ValueError("Insufficient data points for correlation analysis")

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create mask for NaN values to show them in a different color
    mask = corr.isna()

    colors = [
        "#F44336",
        "white",
        "#4CAF50",
        # "#90CAF9",
    ]  # ["darkorange", "gold", "lawngreen", "lightseagreen"]

    cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)

    # Create heatmap
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap=cmap1,
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=1,
        cbar_kws={"label": "Correlation Coefficient"},
        mask=mask,
        ax=ax,
    )

    # Fill in NaN cells with a light gray and "N/A" text
    for i in range(len(metrics)):
        for j in range(len(metrics)):
            if mask.iloc[i, j]:
                ax.add_patch(
                    plt.Rectangle(
                        (j, i),
                        1,
                        1,
                        fill=True,
                        facecolor="lightgray",
                        edgecolor="white",
                        linewidth=1,
                    )
                )
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    "N/A",
                    ha="center",
                    va="center",
                    color="darkgray",
                    fontsize=10,
                )

    # Clean up metric names for display
    cleaned_labels = [_clean_metric_name(m) for m in metrics]
    ax.set_xticklabels(cleaned_labels, rotation=20, ha="center")
    ax.set_yticklabels(cleaned_labels, rotation=0)

    # Styling
    if TITLE:
        ax.set_title("Correlation Between Evaluation Metrics")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    return fig


def plot_metric_boxplots(
    results: Union[EvaluationResult, List[EvaluationResult]],
    metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot box plots showing score distributions for each metric.

    Box plots provide a better view of outliers, quartiles, and overall distribution
    compared to histograms. Shows median, IQR, and outliers for each metric.

    Args:
        results: Single result or list of results to visualize
        metrics: List of metrics to plot. If None, plots all available metrics
        figsize: Figure size (width, height)
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure object

    Raises:
        ValueError: If no valid scores found

    Examples:
        >>> fig = plot_metric_boxplots(results)
        >>> fig = plot_metric_boxplots(results, metrics=['correctness', 'output_relevancy'])
        >>> fig = plot_metric_boxplots(results, save_path='boxplots.png')
    """
    # Convert to DataFrame
    df = results_to_dataframe(results)

    # Get available metrics
    if metrics is None:
        metrics = _get_available_metrics(df)

    if not metrics:
        raise ValueError("No valid metrics found")

    # Exclude NOT APPLICABLE (-1.0) values from distributions
    df = _exclude_not_applicable(df, metrics)

    # Filter to selected metrics
    has_overall = "overall" in metrics
    real_metrics = [m for m in metrics if m != "overall"]

    df_metrics = df[real_metrics]

    if df_metrics.empty:
        raise ValueError("No valid scores found")

    # Build boxplot data, pooling all values for "overall"
    boxplot_data = [df_metrics[m].dropna() for m in real_metrics]
    labels = [_clean_metric_name(m) for m in real_metrics]

    if has_overall:
        pooled = pd.Series(df_metrics.values.flatten()).dropna()
        boxplot_data.append(pooled)
        labels.append(_clean_metric_name("overall"))

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create box plot
    bp = ax.boxplot(
        boxplot_data,
        labels=labels,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="red", markersize=6),
    )

    # Color the boxes
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(DISCRETE_COLORS[i % len(DISCRETE_COLORS)])

    # Styling
    ax.set_ylabel("Score")
    ax.set_title("Score Distribution by Metric (Box Plot)")
    ax.set_ylim(-0.1, 1.1)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="center")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    return fig


def plot_metric_pairplot(
    results: Union[EvaluationResult, List[EvaluationResult]],
    metrics: Optional[List[str]] = None,
    hue: Literal["category", "test_name", "session", "question_variation"] = "category",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot scatter plot matrix (pairplot) showing relationships between metrics.

    Creates a matrix of scatter plots showing how each metric relates to every other
    metric, with distributions on the diagonal. Colored by the specified grouping variable.

    Args:
        results: Single result or list of results to visualize
        metrics: List of metrics to include. If None, uses all available metrics.
                 Recommend 2-4 metrics for readability.
        hue: Column to use for color coding ('category', 'test_name', 'session', 'question_variation')
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure object

    Raises:
        ValueError: If insufficient metrics or no valid data

    Examples:
        >>> fig = plot_metric_pairplot(results)
        >>> fig = plot_metric_pairplot(results, metrics=['correctness', 'output_relevancy'])
        >>> fig = plot_metric_pairplot(results, hue='test_name', save_path='pairplot.png')
    """
    # Convert to DataFrame
    df = results_to_dataframe(results)

    # Get available metrics
    if metrics is None:
        metrics = _get_available_metrics(df)

    if not metrics:
        raise ValueError("No valid metrics found")

    if len(metrics) < 2:
        raise ValueError("Need at least 2 metrics for pairplot")

    # Limit to recommended number for readability
    if len(metrics) > 4:
        print(
            f"Warning: {len(metrics)} metrics selected. "
            "Consider limiting to 4 metrics for better readability."
        )

    # Exclude NOT APPLICABLE (-1.0) values
    df = _exclude_not_applicable(df, metrics)

    # Filter to selected columns and drop NaN
    plot_cols = metrics + [hue]
    df_plot = df[plot_cols].dropna()

    if len(df_plot) < 2:
        raise ValueError("Insufficient data points for pairplot")

    # Create pairplot
    g = sns.pairplot(
        df_plot,
        hue=hue,
        diag_kind="kde",
        plot_kws={"alpha": 0.6, "s": 50},
        diag_kws={"alpha": 0.7},
        palette=DISCRETE_COLORS,
    )

    # Update axis labels to cleaned names
    for i, metric in enumerate(metrics):
        # Update x-axis labels (bottom row)
        g.axes[-1, i].set_xlabel(_clean_metric_name(metric))
        # Update y-axis labels (left column)
        g.axes[i, 0].set_ylabel(_clean_metric_name(metric))

    g.fig.suptitle(
        f"Metric Relationships by {hue.replace('_', ' ').title()}", y=1.02, fontsize=14
    )

    plt.tight_layout()

    if save_path:
        g.savefig(save_path)

    return g.fig


def plot_radar_chart(
    results: Union[EvaluationResult, List[EvaluationResult]],
    metrics: Optional[List[str]] = None,
    group_by: Literal[
        "category", "test_name", "session", "question_variation"
    ] = "test_name",
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot radar/spider chart comparing average performance across all metrics.

    Creates a radar chart showing the average score for each metric across different
    groups (categories, test names, sessions, or question variations). Useful for multi-dimensional comparison.

    Args:
        results: Single result or list of results to visualize
        metrics: List of metrics to include. If None, uses all available metrics
        group_by: Column to group by ('category', 'test_name', 'session', 'question_variation')
        figsize: Figure size (width, height)
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure object

    Raises:
        ValueError: If no valid metrics or insufficient data

    Examples:
        >>> fig = plot_radar_chart(results)
        >>> fig = plot_radar_chart(results, group_by='category')
        >>> fig = plot_radar_chart(results, metrics=['correctness', 'output_relevancy'])
    """
    # Validate group_by
    valid_groupby = ["category", "test_name", "session", "question_variation"]
    if group_by not in valid_groupby:
        raise ValueError(
            f"Invalid group_by: '{group_by}'. Must be one of: {valid_groupby}"
        )

    # Convert to DataFrame
    df = results_to_dataframe(results)

    # Get available metrics
    if metrics is None:
        metrics = _get_available_metrics(df)

    if not metrics:
        raise ValueError("No valid metrics found")

    if len(metrics) < 3:
        raise ValueError("Need at least 3 metrics for meaningful radar chart")

    # Exclude NOT APPLICABLE (-1.0) values from means
    has_overall = "overall" in metrics
    real_metrics = [m for m in metrics if m != "overall"]

    df = _exclude_not_applicable(df, real_metrics)

    # Group by and compute means (NaN values excluded automatically by pandas mean)
    grouped = df.groupby(group_by)[real_metrics].mean()

    if has_overall:
        grouped["overall"] = grouped[real_metrics].mean(axis=1)

    if len(grouped) == 0:
        raise ValueError("No valid data after grouping")

    # Number of variables
    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection="polar"))

    # Plot each group
    for idx, group in enumerate(grouped.index):
        values = grouped.loc[group].tolist()
        values += values[:1]  # Complete the circle
        color = DISCRETE_COLORS[idx % len(DISCRETE_COLORS)]
        ax.plot(
            angles, values, "o-", linewidth=2, label=str(group), color=color, zorder=5
        )
        ax.fill(angles, values, alpha=0.10, color=color, zorder=4)

    # Set ticks but hide default labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])

    # Manually draw metric labels above plot lines
    text_outline = [pe.withStroke(linewidth=1.5, foreground="white")]
    for angle, metric in zip(angles[:-1], metrics):
        ax.text(
            angle,
            1.2,
            _clean_metric_name(metric),
            ha="center",
            va="center",
            size=14,
            zorder=20,
            path_effects=text_outline,
        )

    # Set radial limits and ticks
    ax.set_ylim(0, 1)
    yticks = [0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_yticks(yticks)
    ax.set_yticklabels([])

    # Manually draw radial tick labels above plot lines
    for ytick in yticks:
        ax.text(
            0.2,
            ytick,
            f"{ytick:.1f}",
            ha="center",
            va="bottom",
            size=14,
            zorder=20,
            path_effects=text_outline,
        )

    # Styling
    if TITLE:
        ax.set_title(
            f"Multi-Metric Performance by {group_by.replace('_', ' ').title()}",
            size=14,
            pad=20,
        )
    ax.legend(loc="center left", bbox_to_anchor=(1.3, 0.5))
    ax.grid(True, zorder=1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    return fig
