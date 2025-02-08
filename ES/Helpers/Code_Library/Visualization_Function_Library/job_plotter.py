# job_plotter.py

import matplotlib.pyplot as plt
import numpy as np


class JobPlotter:
    def __init__(self, df, motif_results):
        """
        Initializes the JobPlotter with the data and motif results.

        :param df: DataFrame containing the data to plot.
        :param motif_results: Dictionary containing motif results.
        """
        self.df = df
        self.motif_results = motif_results
        self.jobs = self._define_jobs()
        self.grey_tones = ['#555555', '#bbbbbb', '#cccccc', '#dddddd']  # You can extend this list

    def _define_jobs(self):
        """
        Defines job indices based on the unique job descriptions in the motif results.

        :return: Dictionary mapping job descriptions to indices.
        """
        return {desc: idx for idx, desc in enumerate(set(desc for results in self.motif_results.values()
                                                        for _, _, _, desc in results))}

    def plot(self, highlight_ranges=None):
        """
        Plots the data with the given motif results and highlights specific input ranges.

        :param highlight_ranges: List of tuples (start, end) to highlight specific input ranges.
        """
        fig, axes = plt.subplots(nrows=len(self.motif_results), figsize=(14, 2 * len(self.motif_results)))

        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])  # Ensure axes is always iterable

        for ax, (column, results) in zip(axes, self.motif_results.items()):
            ax.plot(self.df.index[:len(self.df[column].dropna())], self.df[column].dropna() / 1000, label='Data', color='black')
            legend_entries = {column: 'black'}  # Start with the column name as part of legend

            # Highlight motifs
            for start, length, _, job in results:
                color = self.grey_tones[self.jobs[job] % len(self.grey_tones)]
                legend_entries[job] = color
                ax.axvspan(start, start + length, color=color, alpha=0.8, label=job)

            # Highlight specific input ranges in green
            if highlight_ranges:
                for start, end in highlight_ranges:
                    ax.axvspan(start, end, color='green', alpha=0.5, label='Input Highlight')

            # Create custom legend entries based on unique jobs
            custom_legend = [plt.Line2D([0], [0], color=color, lw=4) for desc, color in legend_entries.items()]
            ax.legend(custom_legend, legend_entries.keys(), loc='upper right')
            ax.set_xlabel("Time in s")
            ax.set_ylabel("Active power in kW")

        plt.tight_layout()
        plt.show()
        
