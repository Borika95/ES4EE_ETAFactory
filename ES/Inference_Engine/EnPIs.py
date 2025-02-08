import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import stumpy
import os
import math
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.preprocessing import MinMaxScaler
import stumpy



def calculate_EnPIs(job_dataframes, motif_results, op_counts):
    # Initialize EnPIs dictionary
    EnPIs = {}

    # Initialize other dictionaries
    EnPIs_time = {}
    EnPIs_energy = {}
    job_counts = {}
    job_duration = {}
    job_energy = {}
    productive_time = {}
    productive_energy = {}
    unproductive_time = {}
    unproductive_energy = {}
    unproductive_time_min = {}  # Dictionary to store minimum unproductive time
    unproductive_time_max = {}  # Dictionary to store maximum unproductive time
    UTR = {}  # Dictionary to store the Unproductive Time Ratio
    average_energy_per_job = {}  # Dictionary to store the average energy per job
    average_part_energy_per_job = {}  # Dictionary to store the average part energy per OP XY cycle
    NPTF = {}
    NPEF = {}
    energetic_variance = {}  # Dictionary to store the energetic variance for each job

    # Initialize a dictionary to store energies for each job
    job_energies = {}

    for column, desc_df in job_dataframes.items():
        # Calculate total time and energy for the column
        total_time = len(desc_df) / 3600  # Assume each row represents one second
        time_key = f"Total time {column} in hours"
        EnPIs_time[time_key] = total_time

        total_energy_watts_seconds = desc_df[column].sum()
        total_energy_kWh = total_energy_watts_seconds / (3600 * 1000)
        energy_key = f"Total energy {column} in kWh"
        EnPIs_energy[energy_key] = total_energy_kWh

        # Initialize productive metrics for the column
        productive_time_key = f"Productive time for {column}"
        productive_energy_key = f"Productive energy for {column}"
        productive_time[productive_time_key] = 0
        productive_energy[productive_energy_key] = 0

        # List to hold unproductive periods
        unproductive_durations = []

        # Calculate metrics for each job motif found
        previous_end = 0
        for motif_info in motif_results.get(column, []):
            start, length, _, description = motif_info
            end = start + length

            # Calculate unproductive period before the current motif
            if previous_end < start:
                unproductive_duration = (start - previous_end) / 3600  # Convert seconds to hours
                unproductive_durations.append(unproductive_duration)

            # Job counts and durations
            job_key = f"Number of {description}"
            job_duration_key = f"Time of {description} in hours"
            job_energy_key = f"Energy of {description} in kWh"

            job_counts[job_key] = job_counts.get(job_key, 0) + 1
            duration_hours = length / 3600
            job_duration[job_duration_key] = job_duration.get(job_duration_key, 0) + duration_hours

            # Calculate and accumulate energy
            energy_period = desc_df[column].iloc[start:end].sum()
            energy_period_kWh = energy_period / (3600 * 1000)
            job_energy[job_energy_key] = job_energy.get(job_energy_key, 0) + energy_period_kWh

            # Store energy for each job
            if description not in job_energies:
                job_energies[description] = []
            job_energies[description].append(energy_period_kWh)

            # Accumulate productive time and energy for the column within the loop
            productive_time[productive_time_key] += duration_hours
            productive_energy[productive_energy_key] += energy_period_kWh

            previous_end = end

        # Calculate unproductive period after the last motif
        if previous_end < len(desc_df):
            unproductive_duration = (len(desc_df) - previous_end) / 3600  # Convert seconds to hours
            unproductive_durations.append(unproductive_duration)

        # Calculate unproductive time and energy outside the loop
        unproductive_time_key = f"Unproductive time for {column}"
        unproductive_energy_key = f"Unproductive energy for {column}"
        unproductive_time[unproductive_time_key] = total_time - productive_time[productive_time_key]
        unproductive_energy[unproductive_energy_key] = total_energy_kWh - productive_energy[productive_energy_key]

        # Calculate minimum and maximum unproductive time
        unproductive_time_min_key = f"Minimum unproductive time for {column}"
        unproductive_time_max_key = f"Maximum unproductive time for {column}"

        if unproductive_durations:
            unproductive_time_min[unproductive_time_min_key] = min(unproductive_durations)
            unproductive_time_max[unproductive_time_max_key] = max(unproductive_durations)
        else:
            unproductive_time_min[unproductive_time_min_key] = 0
            unproductive_time_max[unproductive_time_max_key] = 0

        # Calculate UTR (Unproductive Time Ratio)
        UTR_key = f"UTR for {column}"
        if unproductive_time_max[unproductive_time_max_key] > 0:
            UTR[UTR_key] = unproductive_time_min[unproductive_time_min_key] / unproductive_time_max[unproductive_time_max_key]
        else:
            UTR[UTR_key] = 0

        # Calculate NPTF
        NPTF_key = f"NPTF for {column}"
        NPTF[NPTF_key] = unproductive_time[unproductive_time_key] / total_time

        # Calculate NPEF
        NPEF_key = f"NPEF for {column}"
        NPEF[NPEF_key] = unproductive_energy[unproductive_energy_key] / total_energy_kWh

        # Calculate average energy per job cycle for each job type
        for job_key, count in job_counts.items():
            if "Number of " in job_key:
                description = job_key.replace("Number of ", "")
                energy_key = f"Energy of {description} in kWh"
                if job_energy.get(energy_key, 0) > 0 and count > 0:
                    avg_energy_key = f"Average energy per {description} cycle in kWh"
                    average_energy_per_job[avg_energy_key] = job_energy[energy_key] / count

    # Calculate energetic variance for each job type
    for description, energies in job_energies.items():
        if len(energies) > 1:  # More than one energy reading needed to calculate variance
            variance_key = f"Energy variance for {description}"
            energetic_variance[variance_key] = np.var(energies)
        else:
            energetic_variance[f"Energy variance for {description}"] = 0  # Undefined or zero variance

    for op_type, count in op_counts.items():
        op_number = ''.join(filter(str.isdigit, op_type))  # Extract only numeric part from the op_type
        avg_energy_key = f'Average energy per OP {op_number} cycle in kWh'
        if avg_energy_key in average_energy_per_job and count > 0:
            avg_part_energy_key = f"Average part energy per OP {op_number} cycle in kWh"
            average_part_energy_per_job[avg_part_energy_key] = average_energy_per_job[avg_energy_key] / count

    # Combine all results into one dictionary
    EnPIs = {**EnPIs_time, **EnPIs_energy, **job_counts, **job_duration, **job_energy,
             **productive_time, **productive_energy, **unproductive_time, **unproductive_energy,
             **unproductive_time_min, **unproductive_time_max, **UTR,
             **average_energy_per_job, **NPTF, **NPEF, **average_part_energy_per_job, **energetic_variance
            }

    # Normalize specified EnPIs between 0.01 and 0.99
    def normalize_values(values_dict):
        min_val = min(values_dict.values())
        max_val = max(values_dict.values())
        
        # Scale the normalized values to the range [0.01, 0.99]
        return {
            key: 0.01 + (0.99 - 0.01) * ((value - min_val) / (max_val - min_val)) if max_val > min_val else 0.01
            for key, value in values_dict.items()
        }
    
    # Normalize specific EnPIs
    normalized_EnPIs = {}
    normalized_EnPIs.update(normalize_values(NPEF))
    normalized_EnPIs.update(normalize_values(NPTF))
    normalized_EnPIs.update(normalize_values(average_energy_per_job))
    normalized_EnPIs.update(normalize_values(job_counts))
    normalized_EnPIs.update(normalize_values(energetic_variance))
    normalized_EnPIs.update(normalize_values(unproductive_time_min))
    normalized_EnPIs.update(normalize_values(unproductive_time_max))
    normalized_EnPIs.update(normalize_values(UTR))

    # Return job energies separately
    EnPIs['Job Energies'] = job_energies

    # Add normalized EnPIs to the results
    EnPIs['Normalized EnPIs'] = normalized_EnPIs

    return EnPIs, normalized_EnPIs