import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import json
import copy
from sklearn.metrics.pairwise import cosine_similarity
import math
from datetime import datetime # Import datetime for day filtering
import random # For selecting random patients and hours

# --- Import shared functions and config from main pipeline script ---
try:
    # --- MODIFIED: Import from 'oracle.py' ---
    from oracle import (
        load_patient_params,
        get_current_period,
        EVENT_CACHE_PATH,
        PARAM_DIR,
        apps,
        schedules,
        persona_map,
        all_patient_ids
    )
except ImportError:
    print("="*50)
    print("ERROR: Could not import from 'oracle.py'")
    print("Please ensure 'oracle.py' is in the same directory.")
    print("Also ensure 'oracle.py' has an 'if __name__ == \"__main__\":' guard")
    print("to prevent it from running when imported.")
    print("="*50)
    exit(1)
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    exit(1)


# --- New Validation Functions ---

def _get_transitions(event_df):
    """Helper to preprocess event_df and find all app-to-app transitions."""
    df = event_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    df['app'] = df['event'].str.split('_').str[-1]
    df['event_type'] = df['event'].apply(lambda x: 'OPEN' if 'APP_OPEN' in x else ('CLOSE' if 'APP_CLOSE' in x else None))
    df = df[df['event_type'].isin(['OPEN', 'CLOSE'])].copy()
    df = df.sort_values(['patient', 'timestamp'])

    df['next_event_type'] = df.groupby('patient')['event_type'].shift(-1)
    df['next_app'] = df.groupby('patient')['app'].shift(-1)

    # A transition is a CLOSE event followed immediately by an OPEN event
    transitions = df[(df['event_type'] == 'CLOSE') & (df['next_event_type'] == 'OPEN')].copy()
    transitions['hour'] = transitions['timestamp'].dt.hour
    transitions['week'] = transitions['week'].astype(int)
    transitions['date'] = transitions['timestamp'].dt.date

    return transitions

def calculate_hourly_empirical_matrices(transitions_df, patient_id, apps_list):
    """
    Calculates empirical transition matrices for a single patient for each hour (0-23).
    """
    hourly_matrices = {}
    patient_transitions = transitions_df[transitions_df['patient'] == patient_id]

    for hour in range(24):
        hour_transitions = patient_transitions[patient_transitions['hour'] == hour]

        counts = hour_transitions.groupby(['app', 'next_app']).size()
        from_counts = counts.groupby('app').sum()

        # Handle hours with no transitions
        if from_counts.empty:
            hourly_matrices[hour] = pd.DataFrame(0.0, index=apps_list, columns=apps_list)
            continue

        probs = (counts / from_counts).reset_index(name='prob')

        matrix_df = pd.DataFrame(0.0, index=apps_list, columns=apps_list)
        for _, row in probs.iterrows():
            if row['app'] in matrix_df.index and row['next_app'] in matrix_df.columns:
                matrix_df.loc[row['app'], row['next_app']] = row['prob']

        # Ensure all apps are present in the index even if no transitions started from them
        matrix_df = matrix_df.reindex(index=apps_list, columns=apps_list, fill_value=0.0)
        hourly_matrices[hour] = matrix_df.fillna(0)

    return hourly_matrices

def calculate_empirical_matrix_for_period(transitions_df, apps_list, period, schedule):
    """
    Calculates a single empirical transition matrix for a given set of transitions
    and a specific time period.
    """
    # Find all hours that belong to this period
    period_hours = []
    start, end = schedule[period]
    if start <= end:
        period_hours = [h for h in range(start, end)]
    else:
        period_hours = [h for h in range(start, 24)] + [h for h in range(0, end)]

    period_transitions = transitions_df[transitions_df['hour'].isin(period_hours)]

    # Handle periods with no transitions
    if period_transitions.empty:
        return pd.DataFrame(0.0, index=apps_list, columns=apps_list)

    counts = period_transitions.groupby(['app', 'next_app']).size()
    from_counts = counts.groupby('app').sum()

    # Handle periods where transitions exist but 'from_counts' is empty (shouldn't happen, but good check)
    if from_counts.empty:
        return pd.DataFrame(0.0, index=apps_list, columns=apps_list)

    probs = (counts / from_counts).reset_index(name='prob')

    matrix_df = pd.DataFrame(0.0, index=apps_list, columns=apps_list)
    for _, row in probs.iterrows():
        if row['app'] in matrix_df.index and row['next_app'] in matrix_df.columns:
            matrix_df.loc[row['app'], row['next_app']] = row['prob']

    # Ensure all apps are present in the index
    matrix_df = matrix_df.reindex(index=apps_list, columns=apps_list, fill_value=0.0)
    return matrix_df.fillna(0)

def plot_single_patient_dashboard(
    patient_id,
    all_transitions,
    theoretical_params,
    persona_map,
    schedules_map,
    apps_list
):
    """
    Plots the new two-part validation dashboard for a SINGLE patient.
    """
    persona = persona_map[patient_id]
    patient_schedule = schedules_map[persona]
    patient_params = theoretical_params[patient_id]
    patient_transitions = all_transitions[all_transitions['patient'] == patient_id].copy()

    if patient_transitions.empty:
        print(f"Skipping Patient {patient_id} ({persona}): No transition data found.")
        return

    print(f"\n--- Generating Dashboard for Patient {patient_id} ({persona}) ---")

    # --- MODIFIED: Define "Hero" Period for this persona ---
    hero_periods = {
        '9-to-5er': 'Work',
        'Night Owl': 'Recreation',
        'Influencer': 'Active',
        'Compulsive Checker': 'Recreation'
    }
    hero_period = hero_periods.get(persona, list(patient_schedule.keys())[0]) # Default to first period

    # --- Data Prep ---
    # 1. Hourly Empirical Matrices (for top plot)
    hourly_empirical_matrices = calculate_hourly_empirical_matrices(
        patient_transitions, patient_id, apps_list
    )

    # 2. Data slices for bottom plot
    sample_week = 2
    transitions_week_2 = patient_transitions[patient_transitions['week'] == sample_week]

    unique_dates = patient_transitions['date'].unique()
    if len(unique_dates) < 5:
        print(f"Warning: Patient {patient_id} has less than 5 days of data. Using first day.")
        if len(unique_dates) == 0:
             print(f"ERROR: Patient {patient_id} has NO data. Skipping.")
             return
        sample_date = unique_dates[0]
    else:
        sample_date = unique_dates[4] # 5th day
    transitions_day_5 = patient_transitions[patient_transitions['date'] == sample_date]

    # Select 2 random hours for heatmap display
    random_hour_1 = random.randint(0, 23)
    random_hour_2 = random.randint(0, 23)
    while random_hour_2 == random_hour_1: # Ensure they are different
        random_hour_2 = random.randint(0, 23)

    # --- Setup Figure Layout ---
    # --- MODIFIED: 7 rows, 1 column for heatmaps ---
    fig = plt.figure(figsize=(14, 40)) # Narrower figure, but taller
    gs = gridspec.GridSpec(7, 1, # 7 rows, 1 column
                           height_ratios=[3, 1, 1, 1, 1, 1, 1],
                           hspace=1.1, wspace=0.4) # Increased hspace for titles

    # --- Part 1: Top Plot (Hourly Similarity Line Graph) ---
    ax_top = fig.add_subplot(gs[0, 0]) # Span all columns (which is just 1)

    hourly_similarities = []
    for hour in range(24):
        current_period = get_current_period(hour, patient_schedule)
        theo_matrix = patient_params['period_matrices'][current_period]
        emp_matrix = hourly_empirical_matrices.get(hour) # Use .get for safety

        if emp_matrix is None or emp_matrix.sum().sum() == 0:
            similarity = 0.0 # No empirical transitions, similarity is zero
        else:
            theo_vec = theo_matrix.values.flatten().reshape(1, -1)
            emp_vec = emp_matrix.values.flatten().reshape(1, -1)
            similarity = cosine_similarity(theo_vec, emp_vec)[0, 0]
        hourly_similarities.append(similarity)

    # Add vertical lines for period boundaries
    for period, (start, end) in patient_schedule.items():
        ax_top.axvline(start, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        if start < end:
            text_pos = start + (end - start) / 2
        else:
            text_pos = start + (24 - start) / 2

        ax_top.text(text_pos, 0.95, period, transform=ax_top.get_xaxis_transform(),
                    ha='center', va='top', alpha=0.7, fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=0.1))

    ax_top.plot(range(24), hourly_similarities,
                label=f'Patient {patient_id} ({persona})',
                marker='o', markersize=4)

    ax_top.set_title(f"Patient {patient_id} ({persona}): Hour-Level Similarity (Empirical vs. Theoretical)", fontsize=16, pad=20)
    ax_top.set_xlabel("Hour of Day")
    ax_top.set_ylabel("Cosine Similarity (1.0 = Perfect Match)")
    ax_top.set_xticks(range(24))
    ax_top.set_ylim(-0.05, 1.05)
    ax_top.grid(axis='y', linestyle='--', alpha=0.7)
    ax_top.legend(loc='lower right')

    # --- Part 2: Bottom Grid (Per-Period Heatmap Comparison) ---

    # --- MODIFIED: More descriptive row titles ---
    row_titles = [
        f"Theoretical: Ground truth for '{hero_period}' period",
        f"Empirical (All Weeks): Actual accumulated '{hero_period}' behavior",
        f"Empirical (Sample Week {sample_week}): Behavior from one week",
        f"Empirical (Sample Day {sample_date.strftime('%Y-%m-%d')}): Behavior from one day",
        f"Empirical (Sample Hour {random_hour_1}:00): Actual behavior from one random hour",
        f"Empirical (Sample Hour {random_hour_2}:00): Actual behavior from one random hour"
    ]

    data_slices = [
        None, # Placeholder for theoretical
        patient_transitions,
        transitions_week_2,
        transitions_day_5,
        None, # Placeholder for random hour 1
        None  # Placeholder for random hour 2
    ]

    vmax = 1.0
    vmin = 0.0

    # Get the main theoretical matrix for comparison (for rows 0-3)
    theo_hero_matrix = patient_params['period_matrices'][hero_period]

    # --- MODIFIED: Loop for 6 rows, 1 column ---
    for i, (title, data) in enumerate(zip(row_titles, data_slices)):
        ax = fig.add_subplot(gs[i+1, 0]) # Start from row 1, always column 0

        matrix_to_plot = None
        theo_matrix_for_comparison = None

        # Handle all 6 rows
        if i == 0: # Row 0: Theoretical
            matrix_to_plot = theo_hero_matrix
            theo_matrix_for_comparison = theo_hero_matrix # Compare to itself

        elif i == 4: # Row 4: Random Hour 1
            matrix_to_plot = hourly_empirical_matrices.get(random_hour_1)
            hour_1_period = get_current_period(random_hour_1, patient_schedule)
            theo_matrix_for_comparison = patient_params['period_matrices'][hour_1_period]

        elif i == 5: # Row 5: Random Hour 2
            matrix_to_plot = hourly_empirical_matrices.get(random_hour_2)
            hour_2_period = get_current_period(random_hour_2, patient_schedule)
            theo_matrix_for_comparison = patient_params['period_matrices'][hour_2_period]

        else: # Rows 1-3: Empirical Slices (data is a DataFrame)
            theo_matrix_for_comparison = theo_hero_matrix # Compare to the hero period
            if data is None or data.empty:
                matrix_to_plot = None
            else:
                matrix_to_plot = calculate_empirical_matrix_for_period(data, apps_list, hero_period, patient_schedule)

        # --- NEW: Calculate similarity ---
        similarity_str = "(Sim: N/A)"
        if matrix_to_plot is not None and theo_matrix_for_comparison is not None:
            if matrix_to_plot.sum().sum() == 0:
                similarity_str = "(Sim: 0.000)" if theo_matrix_for_comparison.sum().sum() > 0 else "(Sim: 1.000)"
            else:
                vec1 = theo_matrix_for_comparison.values.flatten().reshape(1, -1)
                vec2 = matrix_to_plot.values.flatten().reshape(1, -1)
                similarity = cosine_similarity(vec1, vec2)[0, 0]
                similarity_str = f"(Sim: {similarity:.3f})"
                if i == 0: similarity_str = "(Sim: 1.000)" # By definition
        # --- End NEW ---

        # --- MODIFICATION: Set title ABOVE the heatmap ---
        ax.set_title(f"{title}\n{similarity_str}", fontsize=12, pad=20) # Added similarity string

        # REMOVED: ax.set_ylabel(title, ...)

        # Set Column Title (only for the very first heatmap in this grid)
        if i == 0:
            # This is a sub-title for the whole grid
            ax.text(0.5, 1.35, f"Comparison for Hero Period: '{hero_period}'",
                    fontsize=14, ha='center', transform=ax.transAxes)


        if matrix_to_plot is not None:
            sns.heatmap(matrix_to_plot, ax=ax,
                        cmap='viridis',
                        vmax=vmax, vmin=vmin,
                        annot=True, fmt=".2f", annot_kws={"size": 8},
                        linewidths=.5,
                        cbar=True, # Show colorbar on each
                        xticklabels=apps_list, yticklabels=apps_list)
        else:
             ax.set_xticks([])
             ax.set_yticks([])

        if i == len(row_titles) - 1: # Only on bottom row
            ax.set_xlabel("To App")
        else:
            ax.set_xticklabels([])

        ax.set_ylabel("From App") # Set Y-label for heatmap
        plt.setp(ax.get_xticklabels(), rotation=90)
        plt.setp(ax.get_yticklabels(), rotation=0)


    # --- Save the Dashboard ---
    filename = f"simulation_validation_patient_{patient_id}.png"
    plt.savefig(filename, bbox_inches='tight')
    print(f"Successfully saved validation dashboard to '{filename}'")
    plt.close(fig)


# --- Main execution block ---
if __name__ == "__main__":
    print("--- Starting Simulation Validation ---")

    # --- MODIFIED: Randomly select one hero patient per persona ---
    print("Randomly selecting one hero patient per persona...")
    personas_to_patients = {}
    for patient, persona in persona_map.items():
        if persona not in personas_to_patients:
            personas_to_patients[persona] = []
        personas_to_patients[persona].append(patient)

    hero_patients = []
    for persona, patients in personas_to_patients.items():
        hero_patients.append(random.choice(patients))

    print(f"Selected 'hero' patients: {hero_patients}")
    # --- End Modification ---

    # 2. Load Theoretical Parameters from JSON files
    try:
        theoretical_params = load_patient_params(all_patient_ids, apps, PARAM_DIR)
    except FileNotFoundError as e:
        print(e)
        print(f"Please run 'oracle.py' first to generate {PARAM_DIR} files.")
        exit(1)
    except Exception as e:
        print(f"An error occurred loading params: {e}")
        exit(1)

    # 3. Load Event Log
    if not os.path.exists(EVENT_CACHE_PATH):
        print(f"Event cache '{EVENT_CACHE_PATH}' not found.")
        print(f"Please run 'oracle.py' first to generate the event cache.")
        exit(1)

    print(f"Loading event data from cache: {EVENT_CACHE_PATH}")
    event_df = pd.read_csv(EVENT_CACHE_PATH)
    event_df['timestamp'] = pd.to_datetime(event_df['timestamp'])

    # 4. Pre-calculate all transitions
    print("Preprocessing event log to find all transitions...")
    transitions_df = _get_transitions(event_df)
    if transitions_df.empty:
        print("ERROR: No app-to-app transitions found in the event log.")
        exit(1)

    # 5. Generate a separate dashboard for each hero patient
    for patient_id in hero_patients:
        plot_single_patient_dashboard(
            patient_id,
            transitions_df,
            theoretical_params,
            persona_map,
            schedules,
            apps
        )

    print("\n--- Validation Complete ---")


