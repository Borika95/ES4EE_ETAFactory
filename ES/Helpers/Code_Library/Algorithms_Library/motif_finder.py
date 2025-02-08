import numpy as np
import pandas as pd
import stumpy

class MotifFinder:
    def __init__(self, df):
        self.df = df
        self.patterns = {}
        self.color_map = {}  # Dictionary to manage colors based on job strings

    def add_pattern(self, column, input_start, input_end, input_threshold, job):
        if column not in self.patterns:
            self.patterns[column] = []
        if job not in self.color_map:
            self.color_map[job] = len(self.color_map)
        self.patterns[column].append((input_start, input_end, input_threshold, job))

    def find_motifs(self):
        motif_results = {}
        for column, patterns in self.patterns.items():
            if column not in motif_results:
                motif_results[column] = []
            df_column = self.df[column].dropna().astype(float)

            for input_start, input_end, input_threshold, job in patterns:
                pattern = df_column[input_start:input_end]

                # Define the pre-check segment (first quarter of the pattern)
                quarter_index = int(len(pattern) / 4)
                pre_check_segment = pattern[:quarter_index]

                pattern_mean = pattern.mean()
                pattern_std = pattern.std()
                pre_check_mean = pre_check_segment.mean()
                pre_check_std = pre_check_segment.std()

                profile = stumpy.match(np.asarray(pattern).flatten(), np.asarray(df_column).flatten())
                indices = profile[:, 1]

                for i in indices:
                    motif_end = i + len(pattern)
                    if motif_end > len(df_column):
                        continue
                    motif = df_column[i:motif_end]

                    # Extract the corresponding pre-check segment
                    motif_pre_check_segment = motif[:quarter_index]

                    # Compare the pre-check segment
                    motif_pre_check_mean = motif_pre_check_segment.mean()
                    motif_pre_check_std = motif_pre_check_segment.std()
                    pre_check_mean_diff = abs(motif_pre_check_mean - pre_check_mean) / pre_check_mean
                    pre_check_std_diff = abs(motif_pre_check_std - pre_check_std) / pre_check_std

                    # If the pre-check segment does not match, skip the motif
                    if pre_check_mean_diff > input_threshold or pre_check_std_diff > input_threshold:
                        continue

                    # Proceed to compare the full pattern if the pre-check passes
                    motif_mean = motif.mean()
                    motif_std = motif.std()
                    motif_energy = np.sum(np.square(motif - motif_mean))

                    mean_diff = abs(motif_mean - pattern_mean) / pattern_mean
                    std_diff = abs(motif_std - pattern_std) / pattern_std

                    if mean_diff <= input_threshold and std_diff <= input_threshold:
                        motif_results[column].append((i, len(pattern), self.color_map[job], job, motif_energy))

        # Process results to resolve overlaps
        final_results = self.resolve_overlaps(motif_results)
        return final_results

    def resolve_overlaps(self, motif_results):
        final_results = {}
        for column, results in motif_results.items():
            sorted_results = sorted(results, key=lambda x: x[4], reverse=True)  # Sort by energy
            used_indices = set()
            final_results[column] = []
            for start, length, color_index, job, energy in sorted_results:
                if any(i in used_indices for i in range(start, start + length)):
                    continue
                final_results[column].append((start, length, color_index, job))
                used_indices.update(range(start, start + length))
        return final_results



