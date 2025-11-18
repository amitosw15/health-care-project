import os
import math
import copy
import json
import random as rd
import time  # Added for data generation
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import yaml
import matplotlib
from matplotlib.gridspec import GridSpec


matplotlib.use('Agg')  # <--- Add this line to fix the error

# --- Configuration (loaded only from YAML) ---
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config', 'general.yaml')
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Missing configuration file: {CONFIG_PATH}. Please create it and provide required keys.")

with open(CONFIG_PATH, 'r') as _f:
    _cfg = yaml.safe_load(_f) or {}

def _get(k):
    """Return configuration value for key `k`. Raise if missing (no defaults here)."""
    if k in _cfg:
        return _cfg[k]
    raise KeyError(f"Configuration key '{k}' missing in {CONFIG_PATH}. Add it to the YAML.")

# required top-level config keys
EVENT_CACHE_PATH = _get('event_cache_path')
PARAM_DIR = _get('param_dir')
PERSONA_DIR = _get('persona_dir')
NUM_WEEKS = _get('num_weeks')
PERSONA_MAP = _get('persona_map')
ALL_PATIENT_IDS = list(PERSONA_MAP.keys())
# Removed: HOURS_PER_DAY, DAYS_PER_WEEK (not used)

# optional seed section
_seed_cfg = _cfg.get('seed')
if _seed_cfg:
    if 'numpy' in _seed_cfg:
        np.random.seed(int(_seed_cfg['numpy']))
    if 'random' in _seed_cfg:
        rd.seed(int(_seed_cfg['random']))

# --- Globals loaded from ---
apps = []  # will be set when loading persona params

# --- Experiment parameters (LOADED FROM YAML) ---
SESSION_LENGTH_LAMBDAS = _get('session_length_lambdas')
# Removed: SIMILARITY_LEVELS (logic was removed)

os.makedirs(PARAM_DIR, exist_ok=True)


# --- NEW: Simulation Helper Functions (Self-Contained) ---

def sample_dist(dist_params):
    """Samples a value from a distribution defined in persona_params."""
    dist_type = dist_params.get('type', 'exponential')
    try:
        if dist_type == 'exponential':
            return rd.expovariate(1.0 / dist_params['scale'])
        elif dist_type == 'poisson':
            return np.random.poisson(dist_params['lambda'])
        elif dist_type == 'fixed':
            return dist_params['value']
    except KeyError as e:
        print(f"Error: Missing parameter {e} for distribution {dist_type}")
        return 0
    # Add other distributions as needed
    return 0

def get_next_app(current_app, transition_matrix, marginals):
    """
    Picks the next app based on the transition matrix.
    If current_app is None (session start), picks from marginals.
    """
    if current_app is None:
        # Start of a new session, pick from marginal probabilities
        app_list = list(marginals.keys())
        probs = list(marginals.values())
        return rd.choices(app_list, weights=probs, k=1)[0]

    if current_app not in transition_matrix.index:
        # Fallback: App not in matrix, pick from marginals
        app_list = list(marginals.keys())
        probs = list(marginals.values())
        return rd.choices(app_list, weights=probs, k=1)[0]

    # Transition from the current app
    probs = transition_matrix.loc[current_app].values
    app_list = transition_matrix.columns

    # Ensure probabilities sum to 1
    prob_sum = sum(probs)
    if prob_sum == 0: # No valid transitions
        return None
    normalized_probs = [p / prob_sum for p in probs]

    return rd.choices(app_list, weights=normalized_probs, k=1)[0]


# --- NEW: Main Data Generation Function (Self-Contained) ---

def generate_full_event_stream_sessions(num_weeks, patient_ids, persona_map,
                                        all_persona_params,
                                        session_lambda_override=None,
                                        session_start_override=None):
    """
    Generates a full event stream for all patients.

    This function NOW reads timing_params from each persona.
    The session_lambda_override is used to run experiments.
    """
    global apps # Uses the globally inferred app list
    events = []

    total_seconds = num_weeks * 7 * 24 * 60 * 60
    start_sim_time = time.time() - total_seconds # Start sim `num_weeks` ago

    print(f"Generating {num_weeks} weeks of data for {len(patient_ids)} patients...")

    for patient_id in patient_ids:
        persona_name = persona_map.get(patient_id)
        if not persona_name:
            print(f"Warning: No persona found for patient {patient_id}. Skipping.")
            continue

        persona_params = copy.deepcopy(all_persona_params[persona_name])

        # --- This is the new logic you requested ---
        # 1. Get timing parameters unique to this persona
        timing_params = persona_params.get('timing_params')
        if not timing_params:
            print(f"Warning: No 'timing_params' found for persona '{persona_name}'. Skipping.")
            continue

        # 2. Override the session_length lambda if provided by the experiment
        #    This is the common value for all patients in this run
        if session_lambda_override is not None:
            if 'session_length_dist' in timing_params:
                timing_params['session_length_dist']['lambda'] = session_lambda_override
            else:
                print(f"Warning: 'session_length_dist' not in timing_params for {persona_name}.")

        # --- End of new logic ---
        if session_start_override is not None:
            if 'session_start_dist' in timing_params:
                timing_params['session_start_dist']['value'] = session_start_override
                timing_params['session_start_dist']['type'] = 'fixed'
            else:
                print(f"Warning: 'session_length_dist' not in timing_params for {persona_name}.")

        patient_time = start_sim_time # Each patient starts at the beginning
        session_id_counter = 0

        while patient_time < time.time():
            # 1. How long until the next session?
            try:
                wait_time = sample_dist(timing_params['session_start_dist'])
            except KeyError:
                print(f"Error: 'session_start_dist' missing in timing_params for {persona_name}.")
                break # Stop simulating for this broken persona

            patient_time += wait_time
            if patient_time >= time.time():
                break

            session_id = f"{patient_id}_s{session_id_counter}"
            session_id_counter += 1

            # 2. How many apps in this session?
            try:
                num_events_in_session = sample_dist(timing_params['session_length_dist'])
            except KeyError:
                print(f"Error: 'session_length_dist' missing in timing_params for {persona_name}.")
                num_events_in_session = 1

            if num_events_in_session == 0:
                continue

            current_app = None

            for _ in range(num_events_in_session):
                # 3. Determine period (Work, Night, etc.)
                ts_obj = pd.to_datetime(patient_time, unit='s')
                period_name = get_period_from_timestamp(ts_obj, persona_params.get('day_parts', []))

                if period_name == "Unknown":
                    # No valid period, skip this event
                    continue

                matrices = persona_params['period_matrices']
                marginals = persona_params['period_marginals']

                if period_name not in matrices or period_name not in marginals:
                    # Period defined in day_parts but not in matrices/marginals
                    continue

                # 4. Pick next app
                current_app = get_next_app(current_app,
                                           matrices[period_name],
                                           marginals[period_name])

                if current_app  == "quit":
                    events.append({
                        "timestamp": pd.to_datetime(patient_time, unit='s'),
                        "patient_id": patient_id,
                        "session_id": session_id,
                        "app": current_app,
                        "event_type": "open"
                    })
                    events.append({
                        "timestamp": pd.to_datetime(patient_time, unit='s'),
                        "patient_id": patient_id,
                        "session_id": session_id,
                        "app": current_app,
                        "event_type": "close"
                    })
                    break  # End session on "quit"
                if current_app is None:
                    break # Session ended (e.g., no valid transitions)

                # 5. Add "open" event
                events.append({
                    "timestamp": pd.to_datetime(patient_time, unit='s'),
                    "patient_id": patient_id,
                    "session_id": session_id,
                    "app": current_app,
                    "event_type": "open"
                })

                # 6. How long was the app open?
                try:
                    duration = sample_dist(timing_params['app_duration_dist'])
                except KeyError:
                    print(f"Error: 'app_duration_dist' missing in timing_params for {persona_name}.")
                    duration = 1

                patient_time += duration

                # 7. Add "close" event
                events.append({
                    "timestamp": pd.to_datetime(patient_time, unit='s'),
                    "patient_id": patient_id,
                    "session_id": session_id,
                    "app": current_app,
                    "event_type": "close"
                })

                # 8. How long until next app?
                try:
                    gap = sample_dist(timing_params['inter_event_gap_dist'])
                except KeyError:
                    print(f"Error: 'inter_event_gap_dist' missing in timing_params for {persona_name}.")
                    gap = 0

                patient_time += gap

    print(f"Generation complete. Created {len(events)} events.")
    return pd.DataFrame(events)


# --- CONFIG/PARAM LOADING (Unchanged) ---
def load_persona_params(directory):
    """
    Load persona_*.json from `directory`, reconstruct DataFrames and set global `apps`
    inferred from the first persona matrix. Returns dict persona->params.
    """
    persona_files = [f for f in os.listdir(directory) if f.startswith('persona_') and f.endswith('.json')]
    if not persona_files:
        raise FileNotFoundError(f"No persona JSON files found in '{directory}'. Expected persona_*.json")

    loaded = {}

    for fname in persona_files:
        path = os.path.join(directory, fname)
        with open(path, 'r') as f:
            params = json.load(f)

        # reconstruct matrices as DataFrames
        for period, matrix_dict in params.get('period_matrices', {}).items():
            df = pd.DataFrame.from_dict(matrix_dict, orient='index').fillna(0.0)
            params['period_matrices'][period] = df

        loaded[params.get('persona', fname)] = params

    # assign global apps inferred from persona files (if found)
    global apps
    if not apps:
        apps = list(loaded["9-to-5er"]['period_matrices']["Work"].columns)

    print(f"Loaded {len(loaded)} persona files from '{directory}'. apps inferred: {apps}")
    return loaded

# --- PLOTTING FUNCTIONS (Unchanged) ---
def _plot_matrix_grid(fig, gs_base, period_matrices_dict, apps_list, persona_name, period_list):
    """
    Internal helper function to plot a grid of heatmaps.
    (MODIFIED to remove bar charts and simplify x-axis)
    """
    cols = min(3, len(period_list))

    for idx, period in enumerate(period_list):
        r = idx // cols
        c = idx % cols

        mat = period_matrices_dict.get(period)

        # Plot to the simple grid cell
        ax_heat = fig.add_subplot(gs_base[r, c])

        if mat is None or mat.empty:
            ax_heat.text(0.5, 0.5, f"No data for\n'{period}'", ha='center', va='center', style='italic')
            ax_heat.set_title(f"{persona_name} — {period}")
            ax_heat.set_xticks([])
            ax_heat.set_yticks([])
            continue

        if isinstance(mat, dict):
            df = pd.DataFrame.from_dict(mat, orient='index').reindex(index=apps_list, columns=apps_list).fillna(0.0)
        else:
            df = mat.reindex(index=apps_list, columns=apps_list).fillna(0.0).copy()

        # Normalize rows for heatmap
        row_sums = df.sum(axis=1).replace(0, 1)
        df_norm = df.div(row_sums, axis=0)

        # Calculate marginals (using the non-normalized df)
        col_sums = df.sum(axis=0).reindex(apps_list).fillna(0.0).astype(float)
        total_cols = float(col_sums.sum())
        if total_cols > 0:
            marg_vals = (col_sums / total_cols).tolist()
        else:
            marg_vals = [0.0 for _ in apps_list]

        # heatmap
        sns.heatmap(df_norm, ax=ax_heat, cmap='viridis', vmin=0.0, vmax=1.0, cbar=False, annot=df_norm.size <= 100, fmt=".2f",
                    linewidths=0.3, linecolor='lightgray')

        # --- MODIFICATIONS ---

        # 1. Set new x-ticks to ONLY be the marginal values
        xticks = [f"{v:.2f}" for v in marg_vals]

        # --- FIX IS HERE ---
        # First, define the tick LOCATIONS (at the center of each cell)
        tick_locations = np.arange(len(apps_list)) + 0.5
        ax_heat.set_xticks(tick_locations)
        # --- END FIX ---

        # Now we can safely set the labels to the locations
        ax_heat.set_xticklabels(xticks, rotation=0, ha='center', fontsize=9) # Added fontsize
        ax_heat.set_xlabel("Marginal Probability")
        ax_heat.xaxis.set_label_position('bottom')

        # 2. Set y-ticks (app names) on ALL plots
        ax_heat.set_yticklabels(apps_list, rotation=0)

        ax_heat.set_title(f"{persona_name} — {period}")

        # 3. Bar chart logic is entirely removed.


def plot_and_save_persona_heatmaps(name_tag, persona_params, apps_list, outdir):
    """
    Plots the THEORETICAL matrices from the persona config files.
    """
    os.makedirs(outdir, exist_ok=True)
    fname = os.path.join(outdir, f"{name_tag}_matrices.png")

    periods = list(persona_params['period_matrices'].keys())
    n = len(periods)
    if n == 0:
        return None

    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig_w = cols * 5
    fig_h = rows * 4

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(rows, cols , figure=fig, wspace=0.5, hspace=0.6)

    _plot_matrix_grid(fig, gs,
                      persona_params['period_matrices'],
                      apps_list,
                      persona_params.get('persona', 'Unknown'),
                      periods)

    plt.tight_layout()
    plt.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    return fname

# --- DELETED `run_one_experiment` FUNCTION ---
# Its logic is now in the main block


# --- EMPIRICAL ANALYSIS FUNCTIONS (Unchanged) ---

def get_period_from_timestamp(timestamp, day_parts_list):
    """
    Maps a pandas timestamp to a period name ('Work', 'Night', etc.)
    """
    event_hour = timestamp.hour
    for part in day_parts_list:
        start = part['start_hour']
        end = part['end_hour']
        if start < end:
            if start <= event_hour < end:
                return part['name']
        else:
            if event_hour >= start or event_hour < end:
                return part['name']
    return "Unknown"


def plot_and_save_empirical_heatmaps(name_tag, empirical_matrices, persona_params_map, apps_list, outdir):
    """
    Plots the EMPIRICAL matrices calculated from the event data.
    """
    os.makedirs(outdir, exist_ok=True)
    saved_images = []

    for persona_name, period_matrices_dict in empirical_matrices.items():
        fname = os.path.join(outdir, f"{name_tag}_persona_{persona_name.replace(' ', '_')}_matrices.png")

        if persona_name not in persona_params_map:
            print(f"Warning: Persona '{persona_name}' found in data but not in persona_params_map. Skipping plot.")
            continue

        # Use the THEORETICAL periods to define the plot grid
        # This ensures we plot "No data" for periods that didn't happen
        theoretical_periods = list(persona_params_map[persona_name].get('period_matrices', {}).keys())
        n = len(theoretical_periods)
        if n == 0:
            continue

        cols = min(3, n)
        rows = math.ceil(n / cols)
        fig_w = cols * 5
        fig_h = rows * 4

        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = GridSpec(rows, cols, figure=fig, wspace=0.5, hspace=0.6)

        _plot_matrix_grid(fig, gs,
                          period_matrices_dict,    # The empirical data
                          apps_list,
                          persona_name,
                          theoretical_periods)   # The list of plots to create

        plt.tight_layout()
        plt.savefig(fname, bbox_inches='tight')
        plt.close(fig)
        saved_images.append(fname)

    return saved_images

def calculate_empirical_matrices(event_df, persona_params_map, persona_map, apps_list):
    """
    Calculates the actual, empirical transition matrices from the generated event data.
    """
    if 'event_type' not in event_df.columns:
        print("Warning: 'event_type' column not found. Assuming all events are 'open' events.")
        open_events_df = event_df.copy()
    else:
        open_events_df = event_df[event_df['event_type'].str.lower() == 'open'].copy()

    # --- START DEBUG 1: Print all raw 'open' events in the window ---
    if not open_events_df.empty:
        print(f"\n--- DEBUG: Found {len(open_events_df)} 'open' events in this time window ---")
        debug_df = open_events_df.copy() # Avoid SettingWithCopyWarning
        debug_df['hour'] = debug_df['timestamp'].dt.hour
        # SORTING ADDED HERE
        debug_df = debug_df.sort_values(by=['patient_id', 'timestamp'])
        print("--- All 'open' events (raw data, sorted by patient) ---")
        print(debug_df[['patient_id', 'timestamp', 'hour', 'app', 'session_id']].to_string())
    else:
        print("\n--- DEBUG: No 'open' events found in this time window ---")
    # --- END DEBUG 1 ---

    if open_events_df.empty:
        print("No 'open' events found. Cannot calculate empirical matrices.")
        return {}  # MUST return a dict, not None

    open_events_df['timestamp'] = pd.to_datetime(open_events_df['timestamp'])
    open_events_df['persona'] = open_events_df['patient_id'].map(persona_map)

    def map_period(row):
        persona = row['persona']
        if persona not in persona_params_map:
            return "UnknownPersona"
        day_parts = persona_params_map[persona].get('day_parts', [])
        return get_period_from_timestamp(row['timestamp'], day_parts)

    # 1. Get the period for EVERY open event first
    open_events_df['event_period'] = open_events_df.apply(map_period, axis=1)

    open_events_df = open_events_df.sort_values(by=['patient_id', 'session_id', 'timestamp'])

    # 2. Set the from_app
    open_events_df['from_app'] = open_events_df['app']

    # 3. Get the to_app AND the period of the to_app
    g = open_events_df.groupby(['patient_id', 'session_id'])
    open_events_df['to_app'] = g['app'].shift(-1)
    open_events_df['period'] = g['event_period'].shift(-1)

    # 4. Drop rows that have no destination app (or destination period)
    transitions_df = open_events_df.dropna(subset=['to_app', 'period'])

    # --- START DEBUG 2: Print the transitions used for the plot ---
    if not transitions_df.empty:
        print(f"\n--- DEBUG: Found {len(transitions_df)} transitions to plot ---")
        debug_trans_df = transitions_df.copy() # Avoid SettingWithCopyWarning
        debug_trans_df['hour'] = debug_trans_df['timestamp'].dt.hour
        # SORTING ADDED HERE
        debug_trans_df = debug_trans_df.sort_values(by=['patient_id', 'timestamp'])
        print("--- Transitions used for matrix (from_app -> to_app, sorted by patient) ---")
        # We care about the 'period' of the 'to_app'
        print(debug_trans_df[['patient_id', 'timestamp', 'hour', 'from_app', 'to_app', 'period']].to_string())
        print("------------------------------------------------------\n")
    else:
         print("\n--- DEBUG: No valid transitions found in this window ---")
    # --- END DEBUG 2 ---

    if transitions_df.empty:
        print("No transitions found. Cannot calculate empirical matrices.")
        return {}  # MUST return a dict, not None

    # 5. Now, group by the destination's period (which is correctly named 'period')
    counts = transitions_df.groupby(['persona', 'period', 'from_app', 'to_app']).size().unstack(fill_value=0)

    empirical_matrices = {}
    all_personas = counts.index.get_level_values('persona').unique()

    for persona in all_personas:
        if persona not in empirical_matrices:
            empirical_matrices[persona] = {}

        if persona not in counts.index:
            continue

        persona_counts = counts.loc[persona]
        all_periods = persona_counts.index.get_level_values('period').unique()

        for period in all_periods:
            if period not in persona_counts.index:
                continue

            matrix = persona_counts.loc[period]
            matrix_df = matrix.reindex(index=apps_list, columns=apps_list).fillna(0.0)

            row_sums = matrix_df.sum(axis=1)
            row_sums[row_sums == 0] = 1
            matrix_df = matrix_df.div(row_sums, axis=0)

            empirical_matrices[persona][period] = matrix_df

    return empirical_matrices
# --- MAIN EXECUTION BLOCK (REFACTORED) ---

if __name__ == "__main__":
    results = []

    # 1. Load personas ONCE. This also sets the global `apps` list.
    try:
        all_persona_params = load_persona_params(PERSONA_DIR)
    except FileNotFoundError as e:
        print(e)
        print("Please ensure your persona JSON files (e.g., 'persona_9-to-5er.json') are in the 'params/' directory.")
        exit(1)

    # 2. Plot the THEORETICAL matrices once (they don't change)
    theo_imgs = []
    print("Plotting theoretical matrices from persona files...")
    for persona_name, persona_params_dict in all_persona_params.items():
        name_tag = f"theoretical_persona_{persona_name.replace(' ', '_')}"
        img_path = plot_and_save_persona_heatmaps(name_tag, persona_params_dict, apps, PARAM_DIR)
        if img_path:
            theo_imgs.append(img_path)
    print(f"Saved {len(theo_imgs)} THEORETICAL persona matrix images to {PARAM_DIR}")


    # 3. Loop using parameters from the YAML config
    #    The `run_one_experiment` logic is now inlined here.
    for session_lambda in SESSION_LENGTH_LAMBDAS:
        # Removed the `SIMILARITY_LEVELS` (alpha/tag) loop
        print(f"\n=== RUN: lambda={session_lambda} ===")

        # 3a. Run data generation
        # This function now uses the unique timing params for each persona
        # and applies the session_lambda as an override.
        ev_df = generate_full_event_stream_sessions(
            NUM_WEEKS,
            ALL_PATIENT_IDS,
            PERSONA_MAP,
            all_persona_params,
            session_lambda_override=session_lambda, # This is the experiment variable
            session_start_override=100
        )

        # Save the generated data for this run to a unique file
        run_cache_path = EVENT_CACHE_PATH.replace('.csv', f'_lambda_{session_lambda}.csv')
        ev_df.to_csv(run_cache_path, index=False)
        print(f"Saved events to {run_cache_path} (rows={len(ev_df)})")


        if ev_df.empty:
            print("Event DataFrame is empty, skipping empirical analysis.")
            continue

        # 3b. Run EMPIRICAL analysis on the generated data
        print("Running empirical analysis...")
        ev_df['timestamp'] = pd.to_datetime(ev_df['timestamp'])
        max_time = ev_df['timestamp'].max()

        # --- Analysis for "one week" ---
        one_week_df = ev_df[ev_df['timestamp'] >= max_time - pd.Timedelta(weeks=1)]
        if not one_week_df.empty:
            print(f"Analyzing last week of data ({len(one_week_df)} rows)...")
            empirical_matrices_week = calculate_empirical_matrices(one_week_df, all_persona_params, PERSONA_MAP, apps)
            # Save to a unique file for this run
            emp_imgs_week = plot_and_save_empirical_heatmaps(
                f"empirical_last_week_lambda_{session_lambda}",
                empirical_matrices_week, all_persona_params, apps, PARAM_DIR
            )
            print(f"Saved {len(emp_imgs_week)} EMPIRICAL (1 week) matrix images to {PARAM_DIR}")
        else:
            print("No data in the last week to analyze.")

        # --- Analysis for "one hour" ---
        one_hour_df = ev_df[ev_df['timestamp'] >= max_time - pd.Timedelta(hours=1)]
        if not one_hour_df.empty:
            print(f"Analyzing last hour of data ({len(one_hour_df)} rows)...")
            empirical_matrices_hour = calculate_empirical_matrices(one_hour_df, all_persona_params, PERSONA_MAP, apps)
            # Save to a unique file for this run
            emp_imgs_hour = plot_and_save_empirical_heatmaps(
                f"empirical_last_hour_lambda_{session_lambda}",
                empirical_matrices_hour, all_persona_params, apps, PARAM_DIR
            )
            print(f"Saved {len(emp_imgs_hour)} EMPIRICAL (1 hour) matrix images to {PARAM_DIR}")
        else:
            print("No data in the last hour to analyze.")

        results.append({
            'lambda': session_lambda,
            'images_theoretical': theo_imgs, # These are the same for all runs
            'events_file': run_cache_path,
            'events_rows': len(ev_df)
        })

    # Summarize
    summary_path = os.path.join(PARAM_DIR, "generation_summary.json")
    with open(summary_path, 'w') as sf:
        json.dump(results, sf, indent=2)
    print(f"\nGeneration complete. Summary saved to {summary_path}")