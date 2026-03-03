from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class LangfuseAnalyzer:
    """Generates plots to analyse agent runs."""

    def __init__(self, sessions: List[str]):
        # Prepare the input data once during initialization
        self.df = self._prep_input(sessions)

    def _prep_input(self, sessions: List[str]) -> pd.DataFrame:
        df = pd.concat([pd.read_csv(session) for session in sessions]).reset_index(drop=True)
        df = df[~df["errors"].str.contains("Error code: 429|Error code: 400")]
        df["time_norm"] = self._min_max_normalise(df["time"])
        df["tokens_norm"] = self._min_max_normalise(df["tokens"])
        df["n_errors"] = df["errors"] != "[]"
        return df

    def _min_max_normalise(self, lst: List) -> List:
        min_val = min(lst)
        max_val = max(lst)
        return [(x - min_val) / (max_val - min_val) for x in lst]

    def generate_radar_plot(self, output_path: str = ""):
        """
        Generates a radar plot to compare normalised metrics across different sessions.
        Args:
            df: pandas DataFrame including columns with the session and normalised metrics
            output_path: path for saving the radar plot
        """

        labels = {
            "time_norm": "time",
            "tokens_norm": "tokens",
            "repeated_tool_calls": "repeated\ntool calls",
            "n_errors": "error rate",
        }
        if not self.df["repeated_tool_calls"].any():
            del labels["repeated_tool_calls"]

        all_stats = self.df.groupby("session")[list(labels.keys())].mean()
        data = dict(zip(all_stats.index, all_stats.values))

        num_vars = len(labels)

        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

        for name, values in data.items():
            values = np.append(values, values[0])
            ax.plot(angles, values, label=name)
            ax.fill(angles, values, alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(list(labels.values()))
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

        plt.tight_layout()
        plt.show()
        if output_path:
            plt.savefig(f"{output_path}_radar_plot.png")

    def generate_bar_plot(self, output_path: str = ""):
        """
        Generates a bar plot to compare metrics across different sessions.
        Args:
            df: pandas DataFrame including columns for the session, category,
            test_name and metrics (time, tokens)
            output_path: path for saving the bar plot
        """

        df = (
            self.df.groupby(["session", "category", "test_name"])
            .agg({"time": "mean", "tokens": "mean"})
            .round(2)
            .reset_index()
        )

        fig, axes = plt.subplots(nrows=2, ncols=2, sharex=False, figsize=(10, 8))

        categories = df["category"].unique()
        metrics = ["time", "tokens"]

        for col_idx, category in enumerate(categories):
            df_cat = df[df["category"] == category]
            for row_idx, metric in enumerate(metrics):
                ax = axes[row_idx, col_idx]
                sns.barplot(data=df_cat, x="test_name", hue="session", y=metric, ax=ax)
                ax.set_title(f"{category}")
                ax.set_xlabel("Test name")
                ax.set_ylabel(metric.capitalize())
                if row_idx == 0 and col_idx == 1:
                    handles, labels = ax.get_legend_handles_labels()
                ax.get_legend().remove()
                ax.tick_params(axis="x", labelrotation=90)

        fig.legend(
            handles, labels, title="Session", loc="upper right", bbox_to_anchor=(1.15, 1)
        )

        plt.tight_layout()
        plt.show()
        if output_path:
            plt.savefig(f"{output_path}_bar_plot.png")

    def _create_pie_chart(self, ax, session: str, categories: list, category_colors: dict):
        """Helper method to create a pie chart for a single session."""
        counts = self.df[self.df["session"] == session]["tool_call_correctness"].value_counts()
        # Filter out categories with zero counts
        filtered_categories = [cat for cat in categories if counts.get(cat, 0) > 0]
        ordered_counts = [counts.get(cat, 0) for cat in filtered_categories]
        colors = [category_colors[cat] for cat in filtered_categories]
        ax.pie(
            ordered_counts,
            labels=filtered_categories,
            colors=colors,
            autopct="%1.1f%%",
            startangle=45,
        )
        ax.set_title(session)

    def generate_pie_chart(self, output_path: str = ""):
        """
        Generates pie charts to compare tool call correctness across different sessions.
        Args:
            df: pandas DataFrame including columns for the session and tool_call_correctness
            output_path: path for saving the pie charts
        """
        sessions = self.df["session"].unique()

        color_palette = plt.get_cmap("tab10")
        category_colors = {
            "complete": color_palette(0),  # blue
            "partial": color_palette(2),  # green
            "incorrect": color_palette(1),  # orange
        }
        categories = list(category_colors.keys())

        if len(sessions) == 1:
            fig, ax = plt.subplots(figsize=(4, 4))
            self._create_pie_chart(ax, sessions[0], categories, category_colors)
        else:
            fig, axes = plt.subplots(1, len(sessions), figsize=(len(sessions) * 4, 4))
            for ax, session in zip(axes, sessions):
                self._create_pie_chart(ax, session, categories, category_colors)

        plt.suptitle("Tool call correctness")
        plt.tight_layout()
        plt.show()
        if output_path:
            plt.savefig(f"{output_path}_pie_chart.png")
