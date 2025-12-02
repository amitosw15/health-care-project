import os
import sys
import json
import argparse
import yaml
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib

# --- 1. SETUP MATPLOTLIB BACKEND ---
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# --- 2. MATH IMPORTS ---
from scipy.spatial.distance import mahalanobis, squareform, pdist, jensenshannon
from scipy.linalg import pinv
from scipy.special import rel_entr
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from utils.files import json_matrix_to_numpy, load_full_persona_configs
from js import JSKMeans, compute_robust_js_matrix

# --- 3. CONFIGURATION ---
CONFIG_PATH = 'config/general.yaml'
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Missing config: {CONFIG_PATH}")

with open(CONFIG_PATH, 'r') as _f:
    _cfg = yaml.safe_load(_f) or {}

def _get(k):
    return _cfg.get(k)


def confidence_ellipse(ax, center, cov, scale, **kwargs):
    try:
        lambda_, v = np.linalg.eigh(cov)
        lambda_ = np.sqrt(np.maximum(lambda_, 0))
        ell = Ellipse(xy=center, width=lambda_[0]*scale*2, height=lambda_[1]*scale*2,
                      angle=np.rad2deg(np.arctan2(*v[:, 0][::-1])), **kwargs)
        ax.add_patch(ell)
    except: pass

def calculate_robust_transitions(events_df, time_unit='1min', threshold_minutes=1.5, debug_info=None):
    """
    Core engine to calculate transitions and durations from event data using
    Robust Global Logic (Global Sort + Thresholding).

    Args:
        events_df (pd.DataFrame): Raw event stream.
        time_unit (str): Frequency for exploding intervals (default '1min').
        threshold_minutes (float): Gap size that triggers a 'quit' (default 1.5).
        debug_info (dict, optional): Dictionary with 'patient_id', 'p_type', 'period_name' for debug prints.

    Returns:
        all_transitions (pd.DataFrame): Columns [patient_id, minute_ts, from_app, to_app, hour_ts, hour]
        minute_df (pd.DataFrame): Base minute-level data for marginals [patient_id, minute_ts, app, hour]
    """
    # Determine if we should print debug info for this call
    do_debug = debug_info and \
               debug_info.get('p_type') == 'choose_and_stuck_user'

    if do_debug:
        print(f"\n--- DEBUGGING calculate_robust_transitions for {debug_info['p_type']} ({debug_info['patient_id']}) ---")
        print(f"Initial events_df shape: {events_df.shape}")
        # Filter events_df to the specific patient if needed for more focused debugging
        # (This function already gets events_df for a single patient from visualize_empirical_vs_theoretical)

    # 1. Prepare Data
    df = events_df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['patient_id', 'session_id', 'timestamp'])

    opens = df[df['event_type'] == 'open'].copy()
    closes = df[df['event_type'] == 'close'].copy()

    if do_debug:
        print(f"\n--- After Open/Close extraction ---")
        print(f"opens shape: {opens.shape}")
        print(f"closes shape: {closes.shape}")
        with pd.option_context('display.max_rows', 10):
            if not opens.empty: print("opens head:\n", opens.head())
            if not closes.empty: print("closes head:\n", closes.head())


    # Safety: Align opens/closes
    min_len = min(len(opens), len(closes))
    opens = opens.iloc[:min_len]
    closes = closes.iloc[:min_len]

    if do_debug:
        print(f"\n--- After aligning opens/closes (min_len={min_len}) ---")
        print(f"opens shape: {opens.shape}")
        print(f"closes shape: {closes.shape}")

    # Create Intervals
    intervals = opens[['patient_id', 'timestamp', 'app']].rename(columns={'timestamp': 'start'})
    intervals['end'] = closes['timestamp'].values

    if do_debug:
        print(f"\n--- Intervals created ---")
        print(f"intervals shape: {intervals.shape}")
        with pd.option_context('display.max_rows', 10):
            if not intervals.empty: print("intervals head:\n", intervals.head())

    # Explode to Minute Level
    intervals['minute_range'] = [pd.date_range(s, e, freq=time_unit) for s, e in zip(intervals['start'].dt.floor('min'), intervals['end'].dt.floor('min'))]
    minute_df = intervals.explode('minute_range')[['patient_id', 'minute_range', 'app']].rename(columns={'minute_range': 'minute_ts'})

    # Handle duplicates (rare overlaps)
    minute_df = minute_df.drop_duplicates(subset=['patient_id', 'minute_ts'], keep='last')

    if do_debug:
        print(f"\n--- Minute-level DataFrame (minute_df) ---")
        print(f"minute_df shape: {minute_df.shape}")
        with pd.option_context('display.max_rows', 20):
            if not minute_df.empty: print("minute_df (after explode and drop_duplicates):\n", minute_df) # Show more rows

    # 2. Calculate Gaps GLOBALLY
    minute_df = minute_df.sort_values(['patient_id', 'minute_ts'])
    g = minute_df.groupby('patient_id')

    minute_df['next_app'] = g['app'].shift(-1)
    minute_df['next_ts'] = g['minute_ts'].shift(-1)
    minute_df['prev_ts'] = g['minute_ts'].shift(1)

    # Calculate Gaps (in Minutes)
    minute_df['gap_next'] = (minute_df['next_ts'] - minute_df['minute_ts']).dt.total_seconds() / 60.0
    minute_df['gap_prev'] = (minute_df['minute_ts'] - minute_df['prev_ts']).dt.total_seconds() / 60.0

    if do_debug:
        print(f"\n--- Minute-level DataFrame with Gaps ---")
        with pd.option_context('display.max_rows', 20):
            if not minute_df.empty: print("minute_df (with gaps):\n", minute_df[['patient_id', 'minute_ts', 'app', 'next_app', 'gap_next', 'gap_prev']])

    # 3. Identify Transitions
    is_cont = minute_df['gap_next'] <= threshold_minutes
    is_quit = (minute_df['gap_next'] > threshold_minutes) | (minute_df['gap_next'].isna())
    is_start = (minute_df['gap_prev'] > threshold_minutes) | (minute_df['gap_prev'].isna())

    if do_debug:
        print(f"\n--- Transition Flags ---")
        print(f"is_cont (True if gap <= {threshold_minutes} min): {is_cont.sum()} entries")
        print(f"is_quit (True if gap > {threshold_minutes} min or NaN): {is_quit.sum()} entries")
        print(f"is_start (True if gap > {threshold_minutes} min or NaN): {is_start.sum()} entries")
        
        print("\n--- Quit Details ---")
        with pd.option_context('display.max_rows', 20):
            print(minute_df[is_quit][['minute_ts', 'app', 'gap_next']])


    # 4. Collect Transitions
    transitions_list = []

    # Outgoing (App -> Next / Quit)
    outgoing = minute_df.copy()
    outgoing.loc[is_cont, 'to_app'] = outgoing.loc[is_cont, 'next_app']
    outgoing.loc[is_quit, 'to_app'] = 'quit'
    outgoing['from_app'] = outgoing['app']
    transitions_list.append(outgoing[['patient_id', 'minute_ts', 'from_app', 'to_app']])

    # Incoming (Start -> App)
    incoming = minute_df[is_start].copy()
    incoming['from_app'] = '__start__'
    incoming['to_app'] = incoming['app']
    transitions_list.append(incoming[['patient_id', 'minute_ts', 'from_app', 'to_app']])

    all_transitions = pd.concat(transitions_list)

    if do_debug:
        print(f"\n--- All Transitions Collected ---")
        print(f"all_transitions shape: {all_transitions.shape}")
        with pd.option_context('display.max_rows', 20):
            if not all_transitions.empty: print("all_transitions head:\n", all_transitions) # Show more rows

    # Add time helpers
    all_transitions['hour_ts'] = all_transitions['minute_ts'].dt.floor('h')
    all_transitions['hour'] = all_transitions['minute_ts'].dt.hour
    minute_df['hour'] = minute_df['minute_ts'].dt.hour

    return all_transitions, minute_df

def create_hourly_features(events_df, apps, all_states):
    """
    Generates hourly features (transitions and durations) using robust logic.
    Args:
        events_df: Raw event dataframe
        apps: List of app names (for duration columns)
        all_states: List of all states including 'quit' and '__start__'
    """
    print(f"Running METRONOME featurization... (Loaded {len(events_df)} events)")

    # 1. Ensure special states are in the list
    if 'quit' not in all_states: all_states.append('quit')
    if '__start__' not in all_states: all_states.append('__start__')

    # 2. Call Helper for Robust Transitions
    # This replaces the manual interval explosion and gap calculation
    all_transitions, minute_df = calculate_robust_transitions(events_df)

    # Ensure minute_df has hour_ts for merging later
    minute_df['hour_ts'] = minute_df['minute_ts'].dt.floor('h')

    # 3. Aggregate Transitions by Hour
    all_transitions['trans_col'] = 't_' + all_transitions['from_app'] + '_to_' + all_transitions['to_app']

    # Create the matrix: Rows=Hours, Cols=Transitions
    transition_features = all_transitions.groupby(['patient_id', 'hour_ts', 'trans_col']).size().unstack(fill_value=0)

    # Reindex ensures columns like 't_Work_to_quit' exist even if counts are 0
    expected_cols = [f't_{fr}_to_{to}' for fr in all_states for to in all_states]
    transition_features = transition_features.reindex(columns=expected_cols, fill_value=0).reset_index()

    # 4. Aggregate Durations by Hour
    duration_features = minute_df.groupby(['patient_id', 'hour_ts', 'app']).size().unstack(fill_value=0)
    duration_features = duration_features.reindex(columns=apps).fillna(0)
    duration_features.columns = [f'duration_{app}' for app in duration_features.columns]
    duration_features = duration_features.reset_index()

    # 5. Merge & Formatting
    base_df = minute_df[['patient_id', 'hour_ts']].drop_duplicates()
    features = base_df.merge(duration_features, on=['patient_id', 'hour_ts'], how='left')
    features = features.merge(transition_features, on=['patient_id', 'hour_ts'], how='left').fillna(0)
    features = features.rename(columns={'patient_id': 'patient'})

    features['week'] = features['hour_ts'].dt.to_period('W')
    features['hour'] = features['hour_ts'].dt.hour

    final_cols = [c for c in features.columns if c.startswith('duration_') or c.startswith('t_')]

    print("Metronome featurization complete.")
    return features, final_cols

def visualize_empirical_vs_theoretical(events_df, persona_map, full_configs, output_dir, all_states, dist_metric):
    print("\n--- Generating Comparison Plots (Matrices + Marginals) ---")

    # Ensure special states are in the list for consistent plotting
    display_states = list(all_states)
    if 'quit' not in display_states: display_states.append('quit')
    if '__start__' not in display_states: display_states.append('__start__')

    unique_personas = set(persona_map.values())
    representatives = {}
    for p_type in unique_personas:
        for pid, p_val in persona_map.items():
            if p_val == p_type:
                representatives[p_type] = pid
                break

    for p_type, patient_id in representatives.items():
        if p_type not in full_configs: continue
        p_config = full_configs[p_type]
        schedule = p_config.get('schedule', {})
        theo_matrices = p_config.get('period_matrices', {})
        theo_marginals_dict = p_config.get('period_marginals', {})

        # --- 1. ROBUST METRONOME PREP (Using Helper) ---
        p_events = events_df[events_df['patient_id'] == patient_id].copy()
        if p_events.empty: continue

        # Prepare debug_info for passing to calculate_robust_transitions
        debug_info = {'patient_id': patient_id, 'p_type': p_type, 'period_name': None}

        for period_name, (start_h, end_h) in schedule.items():
            debug_info['period_name'] = period_name # Update period_name for current iteration

            # Call the reusable helper to get clean transitions and minutes
            p_transitions, p_minutes = calculate_robust_transitions(p_events, debug_info=debug_info)
            
            # --- 2. FILTER BY SCHEDULE ---
            # FIX: Use 'hour' (int) column for comparison, not 'hour_ts' (datetime)
            if start_h < end_h:
                mask_trans = (p_transitions['hour'] >= start_h) & (p_transitions['hour'] < end_h)
                mask_mins = (p_minutes['hour'] >= start_h) & (p_minutes['hour'] < end_h)
            else:
                mask_trans = (p_transitions['hour'] >= start_h) | (p_transitions['hour'] < end_h)
                mask_mins = (p_minutes['hour'] >= start_h) | (p_minutes['hour'] < end_h)

            period_trans = p_transitions[mask_trans].copy()
            period_mins = p_minutes[mask_mins].copy()

            if period_mins.empty: continue

            # --- 3. CALCULATE MARGINALS (Time spent in each app) ---
            # Empirical: Simple frequency count of minutes
            emp_marginals = period_mins['app'].value_counts(normalize=True)
            emp_marginals = emp_marginals.reindex(display_states, fill_value=0.0)

            # Theoretical: Load from config
            theo_marginal_vec = pd.Series(0.0, index=display_states)
            if period_name in theo_marginals_dict:
                for app, prob in theo_marginals_dict[period_name].items():
                    if app in theo_marginal_vec:
                        theo_marginal_vec[app] = prob

            # --- 4. CALCULATE EMPIRICAL MATRIX ---
            if not period_trans.empty:
                # Group by (from -> to) using the Robust Transitions
                counts = period_trans.groupby(['from_app', 'to_app']).size().unstack(fill_value=0)
                # Reindex to full square matrix including 'quit' and '__start__'
                counts = counts.reindex(index=display_states, columns=display_states).fillna(0)
                # Normalize
                emp_matrix = counts.div(counts.sum(axis=1), axis=0).fillna(0)
            else:
                emp_matrix = pd.DataFrame(0, index=display_states, columns=display_states)

            # Theoretical Matrix
            if period_name in theo_matrices:
                # Assuming json_matrix_to_numpy is available
                theo_matrix_raw = json_matrix_to_numpy(theo_matrices[period_name], display_states)
                theo_df = pd.DataFrame(theo_matrix_raw, index=display_states, columns=display_states)
            else:
                theo_df = pd.DataFrame(0, index=display_states, columns=display_states)

            # --- 5. PLOTTING ---
            fig, axes = plt.subplots(1, 3, figsize=(24, 7), gridspec_kw={'width_ratios': [1, 1, 0.8]})

            # Panel 1: Empirical Matrix
            sns.heatmap(emp_matrix, annot=True, fmt=".2f", cmap="Reds", cbar=False, ax=axes[0],
                        xticklabels=display_states, yticklabels=display_states)
            axes[0].set_title(f"ACTUAL Matrix\n{p_type} - {period_name}", fontsize=12, fontweight='bold')

            # Panel 2: Theoretical Matrix
            sns.heatmap(theo_df, annot=True, fmt=".2f", cmap="Blues", cbar=False, ax=axes[1],
                        xticklabels=display_states, yticklabels=display_states)
            axes[1].set_title(f"THEORETICAL Matrix\n(Event Choice)", fontsize=12, fontweight='bold')
            axes[1].set_yticks([])

            # Panel 3: Marginals Comparison
            df_marg = pd.DataFrame({
                'State': display_states,
                'Actual (Time)': emp_marginals.values,
                'Theory (Config)': theo_marginal_vec.values
            })

            df_marg_melt = df_marg.melt('State', var_name='Type', value_name='Probability')

            sns.barplot(data=df_marg_melt, x='State', y='Probability', hue='Type', ax=axes[2],
                        palette={'Actual (Time)': 'red', 'Theory (Config)': 'blue'}, alpha=0.7)
            axes[2].set_title(f"Marginal Distribution\n(Time vs Config)", fontsize=12, fontweight='bold')
            axes[2].set_ylim(0, 1.0)
            axes[2].tick_params(axis='x', rotation=45)

            compare_dir = os.path.join(output_dir, "compare_theoretical_empirical", dist_metric)
            os.makedirs(compare_dir, exist_ok=True)
            plt.tight_layout()
            filep = os.path.join(compare_dir, f"compare_{p_type.replace(' ', '_')}_{period_name}.png")
            print(f"Saving comparison plot for persona '{p_type}' during period '{period_name}' to {filep}...")
            plt.savefig(filep)
            plt.close()

def run_analysis_pipeline(train_df, test_df, feature_cols, persona_map, dist_metric='js', visualization_method='tsne', title_prefix=""):
    print(f"\n--- Running Pipeline: {title_prefix} ({dist_metric}) ---")
    try: output_dir = _get('output_dir')
    except KeyError: output_dir = "js_clustering_outputs"
    os.makedirs(output_dir, exist_ok=True)

    try:
        try: config_paths = _get('persona_types_path')
        except KeyError: config_paths = _get('personaa_types_path')
        full_persona_configs = load_full_persona_configs(config_paths)
        persona_schedules = {}
        for p, data in full_persona_configs.items():
            if 'schedule' in data: persona_schedules[p] = data['schedule']
    except KeyError: return

    # --- FEATURE ENGINEERING MODIFICATION ---
    # 1. Select both duration and transition columns for a richer feature set.
    duration_cols = [c for c in feature_cols if c.startswith('duration_')]
    transition_cols = [c for c in feature_cols if c.startswith('t_')]
    combined_feature_cols = duration_cols + transition_cols

    train_X_raw = train_df[combined_feature_cols].values
    test_X_raw = test_df[combined_feature_cols].values

    # 2. Apply StandardScaler to the combined raw counts.
    # This is crucial because duration and transition counts are on different scales.
    # Scaling ensures both feature types contribute fairly to the clustering.
    feature_scaler = StandardScaler()
    train_X_scaled = feature_scaler.fit_transform(train_X_raw)
    # Note: We would use this scaler to transform test_X_raw as well if we were predicting.

    # The rest of the pipeline will now use `train_X_scaled`
    # --- END MODIFICATION ---

    train_labels = pd.Series(train_df['patient'].values).map(persona_map).values

    # Note: The JS distance metric was designed for probability distributions.
    # Since we are now using scaled raw counts, Euclidean distance is more appropriate.
    # The `dist_metric` parameter will be less meaningful unless we add specific logic for it.
    # For now, we proceed with the scaled data, which works well with standard KMeans (Euclidean).

    if visualization_method == 'tsne':
        # t-SNE is often slow. For a quicker feedback loop, especially with many points,
        # running PCA first can be beneficial, but we'll stick to the original logic.
        print("Running t-SNE...")
        # metric='euclidean' is the default and appropriate for our scaled data.
        tsne = TSNE(n_components=2, metric='euclidean', init='random', random_state=42, perplexity=30)
        vis_coords = tsne.fit_transform(train_X_scaled)
    else:
        vis_coords = PCA(n_components=2, random_state=42).fit_transform(train_X_scaled)

    # This scaler is for the 2D visualization, not the clustering itself.
    vis_scaler = StandardScaler()
    vis_coords = vis_scaler.fit_transform(vis_coords)

    def get_label(row):
        p, h = row['persona'], row['hour_ts'].hour
        if p in persona_schedules:
            for lbl, (s, e) in persona_schedules[p].items():
                if (s < e and s <= h < e) or (s > e and (h >= s or h < e)): return f"{p} ({lbl})"
        return f"{p} (Other)"

    k_values = [10, 15]
    all_states = _get('all_states')
    n_states = len(all_states)

    for k in k_values:
        print(f"\n>>> Processing K={k} <<<")
        # K-Means will now run on the scaled, combined features.
        if dist_metric == 'js':
            print("Warning: JS distance is intended for probability distributions. With scaled features, standard KMeans (Euclidean) is used.")
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(train_X_scaled)
        else: # dist_metric == 'euclidean'
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(train_X_scaled)
            
        train_clusters = kmeans.labels_
        h_score = homogeneity_score(train_labels, train_clusters)
        c_score = completeness_score(train_labels, train_clusters)
        v_score = v_measure_score(train_labels, train_clusters)

        plot_df = pd.DataFrame({
            'x': vis_coords[:, 0], 'y': vis_coords[:, 1],
            'cluster': train_clusters, 'patient': train_df['patient'].values,
            'hour_ts': train_df['hour_ts'].values
        })
        plot_df['persona'] = plot_df['patient'].map(persona_map)
        plot_df['style_label'] = plot_df.apply(get_label, axis=1)

        print(f"  Generating cluster breakdown plots...")
        cluster_dir = os.path.join(output_dir, "cluster_analysis")
        os.makedirs(cluster_dir, exist_ok=True)
        dir_pure = os.path.join(cluster_dir, f"k{k}_pure_{dist_metric}")
        dir_mess = os.path.join(cluster_dir, f"k{k}_mess_{dist_metric}")
        os.makedirs(dir_pure, exist_ok=True)
        os.makedirs(dir_mess, exist_ok=True)

        full_data = train_df.copy()
        full_data['cluster'] = train_clusters

        for c_id in range(k):
            c_data = plot_df[plot_df['cluster'] == c_id]
            if c_data.empty: continue
            counts = c_data['style_label'].value_counts(normalize=True)
            top_label = counts.idxmax()
            purity = counts.max()
            target_dir = dir_pure if purity >= 0.95 else dir_mess

            raw_sums = full_data[full_data['cluster'] == c_id][transition_cols].sum(axis=0).values
            total_elements = len(raw_sums)
            if total_elements == n_states * n_states: act_mat = raw_sums.reshape(n_states, n_states)
            elif total_elements % n_states == 0:
                 n_rows = total_elements // n_states
                 partial_mat = raw_sums.reshape(n_rows, n_states)
                 act_mat = np.vstack([partial_mat, np.zeros((n_states - n_rows, n_states))])
            else: continue

            row_sums = act_mat.sum(axis=1, keepdims=True)
            act_mat_norm = np.divide(act_mat, row_sums, out=np.zeros_like(act_mat, dtype=float), where=row_sums!=0)

            top_3 = counts.head(3).index.tolist()
            n_plots = 1 + len(top_3)
            fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
            if n_plots == 1: axes = [axes]
            sns.heatmap(act_mat_norm, annot=True, fmt=".2f", cmap="Reds", cbar=False, ax=axes[0],
                        xticklabels=all_states, yticklabels=all_states)
            axes[0].set_title(f"Cluster {c_id}\nActual Mean\n{top_label} ({purity:.0%})")

            for i, label in enumerate(top_3):
                ax = axes[i+1]
                try:
                    p_name = label.split(" (")[0]
                    t_name = label.split(" (")[1].replace(")", "")
                    if p_name in full_persona_configs and t_name in full_persona_configs[p_name].get('period_matrices', {}):
                        theo_mat = json_matrix_to_numpy(full_persona_configs[p_name]['period_matrices'][t_name], all_states)
                        sns.heatmap(theo_mat, annot=True, fmt=".2f", cmap="Blues", cbar=False, ax=ax,
                                    xticklabels=all_states, yticklabels=all_states)
                        ax.set_title(f"Theoretical:\n{label}")
                    else: ax.text(0.5,0.5,"Config Missing",ha='center')
                except: ax.text(0.5,0.5,"Label Error",ha='center')
            plt.tight_layout()
            plt.savefig(os.path.join(target_dir, f"cluster_{c_id}_breakdown.png"))
            plt.close(fig)

        # SCATTER PLOT
        fig, (ax_plot, ax_txt) = plt.subplots(1, 2, figsize=(24, 10), gridspec_kw={'width_ratios': [3, 1]})
        palette = sns.color_palette('viridis', n_colors=k)
        sns.scatterplot(data=plot_df, x='x', y='y', hue='cluster', style='style_label',
                        s=60, ax=ax_plot, palette=palette, alpha=0.7)
        for c in range(k):
            pts = plot_df[plot_df['cluster']==c][['x','y']].values
            if len(pts) > 2:
                mean = pts.mean(axis=0)
                cov = np.cov(pts, rowvar=False) + np.eye(2)*1e-6
                dists = [mahalanobis(p, mean, pinv(cov)) for p in pts]
                for p_val in np.percentile(dists, [50, 90]):
                    confidence_ellipse(ax_plot, mean, cov, p_val, edgecolor=palette[c], facecolor='none', lw=1)
        ax_plot.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
        ax_txt.axis('off')
        ax_txt.text(0, 0.9, f"K={k}\nH:{h_score:.2f}\nC:{c_score:.2f}\nV:{v_score:.2f}", fontsize=14, family='monospace')
        clusters_dir = os.path.join(output_dir, "clusters_scatter",dist_metric)
        os.makedirs(clusters_dir, exist_ok=True)
        plt.savefig(os.path.join(clusters_dir, f"main_viz_k{k}.png"), bbox_inches='tight')
        plt.close(fig)

if __name__ == "__main__":
    try:
        print("===== STARTING ANALYSIS PIPLINE =====")
        # Caching directory
        CACHE_DIR = ".cache"
        os.makedirs(CACHE_DIR, exist_ok=True)

        file = _get('event_cache_path')
        lambdas = _get('session_length_lambdas')
        distance_metrics = _get('distance_metrics')
        PERSONA_MAP = _get('persona_map')
        apps = _get('apps')
        all_states = _get('all_states')
        for lambda_ in lambdas:
            cur_file = file.replace('.csv', f'_lambda_{lambda_}.csv')
            print(f"Loading event data from file '{cur_file}'...")
            events_df = pd.read_csv(cur_file)

            cache_file = os.path.join(CACHE_DIR, f"hourly_features_lambda_{lambda_}.pkl")
            
            if os.path.exists(cache_file):
                print(f"Loading cached features from '{cache_file}'...")
                import pickle
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                hourly_df = cache_data['hourly_df']
                feature_cols = cache_data['feature_cols']
            else:
                hourly_df, feature_cols = create_hourly_features(events_df,apps, all_states)
                
                print(f"Caching features to '{cache_file}'...")
                import pickle
                with open(cache_file, 'wb') as f:
                    pickle.dump({'hourly_df': hourly_df, 'feature_cols': feature_cols}, f)

            max_week = hourly_df['week'].max()
            train_df = hourly_df[hourly_df['week'] < max_week].copy()
            test_df = hourly_df[hourly_df['week'] == max_week].copy()
            for dist_metric in distance_metrics:
                run_analysis_pipeline(train_df, test_df, feature_cols, PERSONA_MAP, dist_metric, visualization_method=_get('visualization_method'), title_prefix=os.path.basename(file))
                output_dir = _get('output_dir')
                config_paths = _get('persona_types_path')
                full_configs = load_full_persona_configs(config_paths)
                visualize_empirical_vs_theoretical(events_df, PERSONA_MAP, full_configs, output_dir, all_states, dist_metric)
                print(f"\nAnalysis complete for lambda={lambda_}.")
    except Exception as e:
        print(f"Fatal Error: {e}")