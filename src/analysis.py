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

def calculate_robust_transitions(events_df, time_unit='1min', threshold_minutes=1.5):
    """
    Core engine to calculate transitions and durations from event data using
    Robust Global Logic (Global Sort + Thresholding).

    Args:
        events_df (pd.DataFrame): Raw event stream.
        time_unit (str): Frequency for exploding intervals (default '1min').
        threshold_minutes (float): Gap size that triggers a 'quit' (default 1.5).

    Returns:
        all_transitions (pd.DataFrame): Columns [patient_id, minute_ts, from_app, to_app, hour_ts, hour]
        minute_df (pd.DataFrame): Base minute-level data for marginals [patient_id, minute_ts, app, hour]
    """
    # 1. Prepare Data
    df = events_df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['patient_id', 'session_id', 'timestamp'])

    opens = df[df['event_type'] == 'open'].copy()
    closes = df[df['event_type'] == 'close'].copy()

    # Safety: Align opens/closes
    min_len = min(len(opens), len(closes))
    opens = opens.iloc[:min_len]
    closes = closes.iloc[:min_len]

    # Create Intervals
    intervals = opens[['patient_id', 'timestamp', 'app']].rename(columns={'timestamp': 'start'})
    intervals['end'] = closes['timestamp'].values

    # Explode to Minute Level
    intervals['minute_range'] = [pd.date_range(s, e, freq=time_unit) for s, e in zip(intervals['start'].dt.floor('min'), intervals['end'].dt.floor('min'))]
    minute_df = intervals.explode('minute_range')[['patient_id', 'minute_range', 'app']].rename(columns={'minute_range': 'minute_ts'})

    # Handle duplicates (rare overlaps)
    minute_df = minute_df.drop_duplicates(subset=['patient_id', 'minute_ts'], keep='last')

    # 2. Calculate Gaps GLOBALLY
    minute_df = minute_df.sort_values(['patient_id', 'minute_ts'])
    g = minute_df.groupby('patient_id')

    minute_df['next_app'] = g['app'].shift(-1)
    minute_df['next_ts'] = g['minute_ts'].shift(-1)
    minute_df['prev_ts'] = g['minute_ts'].shift(1)

    # Calculate Gaps (in Minutes)
    minute_df['gap_next'] = (minute_df['next_ts'] - minute_df['minute_ts']).dt.total_seconds() / 60.0
    minute_df['gap_prev'] = (minute_df['minute_ts'] - minute_df['prev_ts']).dt.total_seconds() / 60.0

    # 3. Identify Transitions
    is_cont = minute_df['gap_next'] <= threshold_minutes
    # Gap > Threshold OR End of Data (NaN) implies Quit
    is_quit = (minute_df['gap_next'] > threshold_minutes) | (minute_df['gap_next'].isna())
    # Gap > Threshold OR Start of Data (NaN) implies Start
    is_start = (minute_df['gap_prev'] > threshold_minutes) | (minute_df['gap_prev'].isna())

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

        # Call the reusable helper to get clean transitions and minutes
        p_transitions, p_minutes = calculate_robust_transitions(p_events)

        for period_name, (start_h, end_h) in schedule.items():
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

    transition_cols = [c for c in feature_cols if c.startswith('t_')]
    train_X_raw = train_df[transition_cols].values
    test_X_raw = test_df[transition_cols].values
    eps = 1e-9
    train_X_probs = np.divide(train_X_raw + eps, train_X_raw.sum(axis=1, keepdims=True) + eps * train_X_raw.shape[1])
    test_X_probs = np.divide(test_X_raw + eps, test_X_raw.sum(axis=1, keepdims=True) + eps * test_X_raw.shape[1])

    train_labels = pd.Series(train_df['patient'].values).map(persona_map).values

    if visualization_method == 'tsne':
        if dist_metric == 'js':
            if os.path.exists(os.path.join(output_dir, 'js_distance_matrix.npy')):
                print("Loading precomputed JS distance matrix...")
                dist_mat = np.load(os.path.join(output_dir, 'js_distance_matrix.npy'))
            else:
                dist_mat = compute_robust_js_matrix(train_X_probs)
                np.save(os.path.join(output_dir, 'js_distance_matrix.npy'), dist_mat)
        elif dist_metric == 'euclidean':
            dist_mat = squareform(pdist(train_X_probs, metric='euclidean'))
        else:
            print(f"Unknown distance metric: {dist_metric}. Defaulting to Euclidean.")
            dist_mat = squareform(pdist(train_X_probs, metric='euclidean'))
        print("Running t-SNE...")
        vis_coords = TSNE(n_components=2, metric='precomputed', init='random', random_state=42).fit_transform(dist_mat)
    else:
        vis_coords = PCA(n_components=2, random_state=42).fit_transform(train_X_probs)

    scaler = StandardScaler()
    vis_coords = scaler.fit_transform(vis_coords)

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
        if dist_metric == 'euclidean':
            kmeans = KMeans(n_clusters=k, random_state=42).fit(train_X_probs)
        elif dist_metric == 'js':
            kmeans = JSKMeans(n_clusters=k, random_state=42).fit(train_X_probs)
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
            hourly_df, feature_cols = create_hourly_features(events_df,apps, all_states)
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