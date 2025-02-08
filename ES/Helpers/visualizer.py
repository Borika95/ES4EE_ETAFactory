import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm  # For colormap

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




class JobPlotterColored:
    def __init__(self, df, motif_results):
        """
        Initializes the JobPlotterColored with the data and motif results.

        :param df: DataFrame containing the data to plot.
        :param motif_results: Dictionary containing motif results.
        """
        # Set global font properties
        rcParams['font.family'] = 'Palatino Linotype'
        rcParams['font.size'] = 20

        self.df = df
        self.motif_results = motif_results
        self.jobs = self._define_jobs()
        self.colors = plt.cm.Spectral  # Use the Spectral colormap

    def _define_jobs(self):
        """
        Defines job indices based on the unique job descriptions in the motif results.

        :return: Dictionary mapping job descriptions to indices.
        """
        return {desc: idx for idx, desc in enumerate(set(desc for results in self.motif_results.values()
                                                        for _, _, _, desc in results))}

    def _get_color(self, job):
        """
        Gets the color for a specific job based on its index.

        :param job: Job description.
        :return: RGBA color from the colormap.
        """
        return self.colors(self.jobs[job] / (len(self.jobs) - 1))  # Normalize to [0, 1] range


    def plot(self, highlight_ranges=None, filename=None):
        """
        Plots the data with the given motif results and highlights specific input ranges.
    
        :param highlight_ranges: List of tuples (start, end) to highlight specific input ranges.
        :param filename: Optional. If provided, saves the plot to the specified file (e.g., "output.svg").
        """
        fig, axes = plt.subplots(nrows=len(self.motif_results), figsize=(15, 10))  # Removed sharey=True
    
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])  # Ensure axes is always iterable
    
        for i, (ax, (column, results)) in enumerate(zip(axes, self.motif_results.items())):
            # Plot the main data
            data_to_plot = self.df[column].dropna() / 1000
            ax.plot(self.df.index[:len(data_to_plot)], data_to_plot, label='Data', color='black')
            legend_entries = {column: 'black'}  # Start with the column name as part of legend
    
            # Highlight motifs
            for start, length, _, job in results:
                color = self._get_color(job)
                legend_entries[job] = color
                ax.axvspan(start, start + length, color=color, alpha=0.8, label=job)
    
            # Highlight specific input ranges in green
            if highlight_ranges:
                for start, end in highlight_ranges:
                    ax.axvspan(start, end, color='green', alpha=0.5, label='Input Highlight')
    
            # Set x-axis limits
            ax.set_xlim(0, 14000)
    
            # Autoscale the y-axis based on the data in the subplot
            ax.autoscale(axis='y')
    
            # Create custom legend entries based on unique jobs
            custom_legend = [plt.Line2D([0], [0], color=color, lw=4) for desc, color in legend_entries.items()]
            ax.legend(custom_legend, legend_entries.keys(), loc='upper right', fontsize=20)  # Increase legend font size
    
            # Set x-axis label and ticks only for the lowest subplot
            if i == len(axes) - 1:
                ax.set_xlabel("Time in s", fontsize=20)  # Increase x-axis label font size
            else:
                ax.tick_params(labelbottom=False)  # Hide x-axis ticks for all but the last subplot
    
        # Set a single shared y-axis label
        fig.text(0.04, 0.5, "Active power in kW", va='center', ha='center', rotation='vertical', fontsize=20)  # Increase y-axis label font size
    
        plt.tight_layout(rect=[0.05, 0, 1, 1])  # Adjust layout to accommodate shared y-axis label
    





class FuzzyVisualizer:
    def __init__(self):
        rcParams['font.family'] = 'Palatino Linotype'
        rcParams['font.size'] = 9
        plt.rcParams['hatch.linewidth'] = 1.5

    def visualize_fuzzy_variable(self, fuzzy_var, xlabel, ylabel):
        # Use the 'viridis' colormap
        cmap = cm.get_cmap('viridis')
        
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.tight_layout(pad=5.0)
        legend_patches = []
        
        # Get a set of unique colors from the colormap
        colors = [cmap(i / len(fuzzy_var.terms)) for i in range(len(fuzzy_var.terms))]

        for i, term in enumerate(fuzzy_var.terms):
            universe = fuzzy_var.universe
            mf_values = fuzzy_var[term].mf
            
            # Use the viridis colormap for the lines and filled areas
            color = colors[i % len(colors)]
            ax.plot(universe, mf_values, color=color, linewidth=2)
            ax.fill_between(universe, 0, mf_values, color=color, alpha=0.7)
            patch = mpatches.Patch(color=color, label=term)
            legend_patches.append(patch)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.legend(handles=legend_patches, loc='best', handleheight=2.5, handlelength=2)
        
        return fig

    def visualize_fuzzy_rules(self, control_system_simulation, input_labels, output_label):
        """
        Visualize fuzzy rules. Can handle both 2D and 3D visualization based on the number of inputs.
        
        Args:
        - control_system_simulation: Fuzzy control system simulation object.
        - input_labels: List of input variable labels. Should be 2 or 3 inputs.
        - output_label: Output variable label.
        """
        num_inputs = len(input_labels)
        
        if num_inputs == 2:
            # Generate input value grids for 2D plot
            input1_values = np.linspace(0.01, 0.99, 50)  # Avoid 0 and 1
            input2_values = np.linspace(0.01, 0.99, 50)
            input1_grid, input2_grid = np.meshgrid(input1_values, input2_values)
            output_grid = np.zeros_like(input1_grid)
    
            # Simulate the grid
            for i in range(input1_grid.shape[0]):
                for j in range(input1_grid.shape[1]):
                    control_system_simulation.input[input_labels[0]] = input1_grid[i, j]
                    control_system_simulation.input[input_labels[1]] = input2_grid[i, j]
                    try:
                        control_system_simulation.compute()
                        output_grid[i, j] = control_system_simulation.output[output_label]
                    except Exception as e:
                        print(f"Error at grid point ({i}, {j}): {e}")
                        output_grid[i, j] = np.nan  # Handle errors by setting NaN values
    
            # Create the 2D heatmap using the 'viridis' colormap
            fig, ax = plt.subplots()
            c = ax.pcolormesh(input1_grid, input2_grid, output_grid, cmap='viridis', shading='auto')
            ax.set_xlabel(input_labels[0])
            ax.set_ylabel(input_labels[1])
    
            # Add a color bar to represent the scale
            colorbar = fig.colorbar(c, ax=ax)
            colorbar.set_label(output_label)
    
        elif num_inputs == 3:
            # Generate input value grids for 3D plot
            input1_values = np.linspace(0.01, 0.99, 25)
            input2_values = np.linspace(0.01, 0.99, 25)
            input3_values = np.linspace(0.01, 0.99, 25)
            input1_grid, input2_grid, input3_grid = np.meshgrid(input1_values, input2_values, input3_values)
            output_grid = np.zeros_like(input1_grid)
    
            # Simulate the grid
            for i in range(input1_grid.shape[0]):
                for j in range(input2_grid.shape[1]):
                    for k in range(input3_grid.shape[2]):
                        control_system_simulation.input[input_labels[0]] = input1_grid[i, j, k]
                        control_system_simulation.input[input_labels[1]] = input2_grid[i, j, k]
                        control_system_simulation.input[input_labels[2]] = input3_grid[i, j, k]
                        try:
                            control_system_simulation.compute()
                            output_grid[i, j, k] = control_system_simulation.output[output_label]
                        except Exception as e:
                            print(f"Error at grid point ({i}, {j}, {k}): {e}")
                            output_grid[i, j, k] = np.nan  # Handle errors by setting NaN values
    
            # Create the 3D surface plot using the 'viridis' colormap
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(input1_grid[:, :, 0], input2_grid[:, :, 0], output_grid[:, :, 0], cmap='viridis')
    
            # Add labels
            ax.set_xlabel(input_labels[0])
            ax.set_ylabel(input_labels[1])
            ax.set_zlabel(output_label)
    
        else:
            raise ValueError("Only 2 or 3 input variables are supported.")
    
        return fig