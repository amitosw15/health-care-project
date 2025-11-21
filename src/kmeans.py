import os
# --- VAE WORKAROUND REMOVED ---

import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse
from scipy.spatial.distance import mahalanobis
from scipy.linalg import pinv

# --- TENSORFLOW/VAE IMPORTS REMOVED ---

# Scikit-learn imports
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# UMAP: Requires 'pip install umap-learn'
try:
    import umap.umap_ as umap
except ImportError:
    print("Warning: UMAP not installed. 'pip install umap-learn'. UMAP visualization will not be available.")
    umap = None

CONFIG_PATH = os.path.join(os.getcwd(), 'config', 'general.yaml')
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Missing configuration file: {CONFIG_PATH}.")

with open(CONFIG_PATH, 'r') as _f:
    _cfg = yaml.safe_load(_f) or {}

def _get(k):
    """Return configuration value for key `k`. Raise if missing."""
    if k in _cfg:
        return _cfg[k]
    raise KeyError(f"Configuration key '{k}' missing in {CONFIG_PATH}. Add it to the YAML.")



import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
from scipy.spatial.distance import mahalanobis, squareform, pdist
from scipy.linalg import pinv
from scipy.special import rel_entr
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# --- HELPERS ---

def json_matrix_to_numpy(period_matrix_dict, all_states):
    """Converts JSON dict (row->col->prob) to sorted Numpy array."""
    n = len(all_states)
    matrix = np.zeros((n, n))
    state_map = {s.lower(): i for i, s in enumerate(all_states)}

    for row_name, probs in period_matrix_dict.items():
        r_idx = state_map.get(row_name.lower())
        if r_idx is None: continue
        for col_name, prob in probs.items():
            c_idx = state_map.get(col_name.lower())
            if c_idx is not None:
                matrix[r_idx, c_idx] = prob
    return matrix

def load_full_persona_configs(config_paths):
    """Loads full JSON content."""
    configs = {}
    print("Loading full persona configurations...")
    for persona_name, file_path in config_paths.items():
        file_path = file_path.strip()
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    configs[persona_name] = json.load(f)
            except Exception as e:
                print(f"  [Error] {file_path}: {e}")
    return configs

def compute_robust_js_matrix(X):
    """Computes N x N Jensen-Shannon distance matrix."""
    print(f"Computing JS distance matrix for {X.shape[0]} samples...")
    def js_metric(p, q):
        p = p.reshape(1, -1)
        q = q.reshape(1, -1)
        m = 0.5 * (p + q)
        left = rel_entr(p, m).sum()
        right = rel_entr(q, m).sum()
        return np.sqrt(max(0.5 * (left + right), 0.0))
    return squareform(pdist(X, metric=js_metric))

class JSKMeans:
    """Custom K-Means for Probability Distributions."""
    def __init__(self, n_clusters=5, max_iter=300, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None

    def fit(self, X):
        np.random.seed(self.random_state)
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]

        for i in range(self.max_iter):
            dists = self._calc_dists(X, self.centroids)
            labels = np.argmin(dists, axis=1)
            new_centroids = np.zeros_like(self.centroids)
            for k in range(self.n_clusters):
                mask = (labels == k)
                if np.any(mask): new_centroids[k] = X[mask].mean(axis=0)
                else: new_centroids[k] = self.centroids[k]

            if np.linalg.norm(self.centroids - new_centroids) < self.tol: break
            self.centroids = new_centroids
            self.labels_ = labels
        return self

    def predict(self, X):
        return np.argmin(self._calc_dists(X, self.centroids), axis=1)

    def _calc_dists(self, X, centroids):
        n, k = X.shape[0], centroids.shape[0]
        dists = np.zeros((n, k))
        for i in range(k):
            p, q = X, centroids[i].reshape(1, -1)
            m = 0.5 * (p + q)
            js = 0.5 * (rel_entr(p, m).sum(axis=1) + rel_entr(q, m).sum(axis=1))
            dists[:, i] = np.sqrt(np.maximum(js, 0.0))
        return dists

def confidence_ellipse(ax, center, cov, scale, **kwargs):
    try:
        lambda_, v = np.linalg.eigh(cov)
        lambda_ = np.sqrt(np.maximum(lambda_, 0))
        ell = Ellipse(xy=center, width=lambda_[0]*scale*2, height=lambda_[1]*scale*2,
                      angle=np.rad2deg(np.arctan2(*v[:, 0][::-1])), **kwargs)
        ax.add_patch(ell)
    except: pass

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
from scipy.spatial.distance import mahalanobis, squareform, pdist
from scipy.linalg import pinv
from scipy.special import rel_entr
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# --- HELPERS ---

def json_matrix_to_numpy(period_matrix_dict, all_states):
    """Converts JSON dict (row->col->prob) to sorted Numpy array."""
    n = len(all_states)
    matrix = np.zeros((n, n))
    state_map = {s.lower(): i for i, s in enumerate(all_states)}

    for row_name, probs in period_matrix_dict.items():
        r_idx = state_map.get(row_name.lower())
        if r_idx is None: continue
        for col_name, prob in probs.items():
            c_idx = state_map.get(col_name.lower())
            if c_idx is not None:
                matrix[r_idx, c_idx] = prob
    return matrix

def load_full_persona_configs(config_paths):
    """Loads full JSON content."""
    configs = {}
    print("Loading full persona configurations...")
    for persona_name, file_path in config_paths.items():
        file_path = file_path.strip()
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    configs[persona_name] = json.load(f)
            except Exception as e:
                print(f"  [Error] {file_path}: {e}")
    return configs

def compute_robust_js_matrix(X):
    """Computes N x N Jensen-Shannon distance matrix."""
    print(f"Computing JS distance matrix for {X.shape[0]} samples...")
    def js_metric(p, q):
        p = p.reshape(1, -1)
        q = q.reshape(1, -1)
        m = 0.5 * (p + q)
        left = rel_entr(p, m).sum()
        right = rel_entr(q, m).sum()
        return np.sqrt(max(0.5 * (left + right), 0.0))
    return squareform(pdist(X, metric=js_metric))

class JSKMeans:
    """Custom K-Means for Probability Distributions."""
    def __init__(self, n_clusters=5, max_iter=300, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None

    def fit(self, X):
        np.random.seed(self.random_state)
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]

        for i in range(self.max_iter):
            dists = self._calc_dists(X, self.centroids)
            labels = np.argmin(dists, axis=1)
            new_centroids = np.zeros_like(self.centroids)
            for k in range(self.n_clusters):
                mask = (labels == k)
                if np.any(mask): new_centroids[k] = X[mask].mean(axis=0)
                else: new_centroids[k] = self.centroids[k]

            if np.linalg.norm(self.centroids - new_centroids) < self.tol: break
            self.centroids = new_centroids
            self.labels_ = labels
        return self

    def predict(self, X):
        return np.argmin(self._calc_dists(X, self.centroids), axis=1)

    def _calc_dists(self, X, centroids):
        n, k = X.shape[0], centroids.shape[0]
        dists = np.zeros((n, k))
        for i in range(k):
            p, q = X, centroids[i].reshape(1, -1)
            m = 0.5 * (p + q)
            js = 0.5 * (rel_entr(p, m).sum(axis=1) + rel_entr(q, m).sum(axis=1))
            dists[:, i] = np.sqrt(np.maximum(js, 0.0))
        return dists

def confidence_ellipse(ax, center, cov, scale, **kwargs):
    try:
        lambda_, v = np.linalg.eigh(cov)
        lambda_ = np.sqrt(np.maximum(lambda_, 0))
        ell = Ellipse(xy=center, width=lambda_[0]*scale*2, height=lambda_[1]*scale*2,
                      angle=np.rad2deg(np.arctan2(*v[:, 0][::-1])), **kwargs)
        ax.add_patch(ell)
    except: pass

# --- MAIN ANALYSIS PIPELINE ---

def run_analysis_pipeline(train_df, test_df, feature_cols, persona_map, visualization_method='tsne', title_prefix=""):
    print(f"\n--- Running Pipeline: {title_prefix} (JS Distance + Pure/Mess Analysis) ---")

    # 0. Setup Output
    try: output_dir = _get('output_dir')
    except KeyError: output_dir = "analysis_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Load Configs
    try:
        try: config_paths = _get('persona_types_path')
        except KeyError: config_paths = _get('personaa_types_path')
        full_persona_configs = load_full_persona_configs(config_paths)
        persona_schedules = {}
        for p, data in full_persona_configs.items():
            if 'schedule' in data: persona_schedules[p] = data['schedule']
    except KeyError:
        print("Error: Missing persona config paths.")
        return

    # 1. Data Prep (Transitions Only)
    transition_cols = [c for c in feature_cols if c.startswith('t_')]
    train_X_raw = train_df[transition_cols].values
    test_X_raw = test_df[transition_cols].values

    # Normalize
    eps = 1e-9
    train_X_probs = np.divide(train_X_raw + eps, train_X_raw.sum(axis=1, keepdims=True) + eps * train_X_raw.shape[1])
    test_X_probs = np.divide(test_X_raw + eps, test_X_raw.sum(axis=1, keepdims=True) + eps * test_X_raw.shape[1])

    train_labels = pd.Series(train_df['patient'].values).map(persona_map).values
    test_labels = pd.Series(test_df['patient'].values).map(persona_map).values

    # 2. Visualization Coordinates
    if visualization_method == 'tsne':
        dist_mat = compute_robust_js_matrix(train_X_probs)
        print("Running t-SNE (Precomputed)...")
        vis_coords = TSNE(n_components=2, metric='precomputed', init='random', random_state=42).fit_transform(dist_mat)
    else:
        vis_coords = PCA(n_components=2, random_state=42).fit_transform(train_X_probs)

    def get_label(row):
        p, h = row['persona'], row['hour_ts'].hour
        if p in persona_schedules:
            for lbl, (s, e) in persona_schedules[p].items():
                if (s < e and s <= h < e) or (s > e and (h >= s or h < e)): return f"{p} ({lbl})"
        return f"{p} (Other)"

    # 3. Loop K
    k_values = [5, 10, 15, 20, 25]
    all_states = _get('all_states')
    n_states = len(all_states)

    for k in k_values:
        print(f"\n>>> Processing K={k} <<<")

        js_kmeans = JSKMeans(n_clusters=k, random_state=42).fit(train_X_probs)
        train_clusters = js_kmeans.labels_
        test_clusters = js_kmeans.predict(test_X_probs)

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

        # --- A. CLUSTER ANALYSIS (PURE vs MESS) ---
        print(f"  Generating cluster breakdown plots...")
        dir_pure = os.path.join(output_dir, f"k{k}_pure")
        dir_mess = os.path.join(output_dir, f"k{k}_mess")
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

            is_pure = purity >= 0.95
            target_dir = dir_pure if is_pure else dir_mess
            status_str = "PURE" if is_pure else "MESS"

            # 3. Calculate ACTUAL Mean Matrix (Red)
            # Sum raw transitions
            raw_sums = full_data[full_data['cluster'] == c_id][transition_cols].sum(axis=0).values

            # --- FIX: Handle Reshape Robustly ---
            total_elements = len(raw_sums)
            # Check if square (6x6=36)
            if total_elements == n_states * n_states:
                 act_mat = raw_sums.reshape(n_states, n_states)
            # Check if rectangular (5x6=30) -> Pad to square
            elif total_elements % n_states == 0:
                 n_rows = total_elements // n_states # Likely 5
                 partial_mat = raw_sums.reshape(n_rows, n_states)
                 # Pad missing rows with zeros
                 missing_rows = n_states - n_rows
                 act_mat = np.vstack([partial_mat, np.zeros((missing_rows, n_states))])
            else:
                 print(f"Error: Vector size {total_elements} does not fit n_states={n_states}")
                 continue
            # ------------------------------------

            # Row Normalize
            row_sums = act_mat.sum(axis=1, keepdims=True)
            act_mat_norm = np.divide(act_mat, row_sums, out=np.zeros_like(act_mat), where=row_sums!=0)

            # 4. Setup Plot
            top_3 = counts.head(3).index.tolist()
            n_plots = 1 + len(top_3)
            fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
            if n_plots == 1: axes = [axes]

            # Plot Actual
            sns.heatmap(act_mat_norm, annot=True, fmt=".2f", cmap="Reds", cbar=False,
                        xticklabels=all_states, yticklabels=all_states, ax=axes[0])
            axes[0].set_title(f"Cluster {c_id} ({status_str})\nActual Mean\nDominant: {top_label} ({purity:.0%})")

            # Plot Theoreticals (Blue)
            for i, label in enumerate(top_3):
                ax = axes[i+1]
                try:
                    p_name = label.split(" (")[0]
                    t_name = label.split(" (")[1].replace(")", "")
                    if p_name in full_persona_configs:
                        matrices = full_persona_configs[p_name].get('period_matrices', {})
                        if t_name in matrices:
                            theo_mat = json_matrix_to_numpy(matrices[t_name], all_states)
                            sns.heatmap(theo_mat, annot=True, fmt=".2f", cmap="Blues", cbar=False,
                                        xticklabels=all_states, yticklabels=all_states, ax=ax)
                            ax.set_title(f"Theoretical:\n{label}\n({counts[label]:.0%} of cluster)")
                        else: ax.text(0.5,0.5,"Matrix Missing",ha='center')
                    else: ax.text(0.5,0.5,"Config Missing",ha='center')
                except: ax.text(0.5,0.5,"Label Parse Error",ha='center')

            plt.tight_layout()
            fname = os.path.join(target_dir, f"cluster_{c_id}_breakdown.png")
            plt.savefig(fname)
            plt.close()

        # --- B. MAIN SCATTER PLOT ---
        fig, (ax_plot, ax_txt) = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw={'width_ratios': [3, 1]})
        palette = sns.color_palette('viridis', n_colors=k)

        sns.scatterplot(data=plot_df, x='x', y='y', hue='cluster', style='style_label',
                        s=60, ax=ax_plot, palette=palette, alpha=0.7, legend=False)

        # Quantiles
        cluster_means = {c: plot_df[plot_df['cluster']==c][['x','y']].mean() for c in range(k)}
        for c in range(k):
            pts = plot_df[plot_df['cluster']==c][['x','y']].values
            if len(pts) > 2:
                mean = pts.mean(axis=0)
                cov = np.cov(pts, rowvar=False) + np.eye(2)*1e-6
                dists = [mahalanobis(p, mean, pinv(cov)) for p in pts]
                for p_val in np.percentile(dists, [50, 90]):
                    confidence_ellipse(ax_plot, mean, cov, p_val, edgecolor=palette[c], facecolor='none', lw=1)

        ax_plot.set_title(f"K={k} | JS Distance Clustering", fontsize=16)
        ax_txt.axis('off')
        ax_txt.text(0, 0.9, f"K={k}\nHomogeneity: {h_score:.2f}\nCompleteness: {c_score:.2f}\nV-Measure: {v_score:.2f}", fontsize=14, family='monospace')

        plt.savefig(os.path.join(output_dir, f"main_viz_k{k}.png"), bbox_inches='tight')
        plt.close()
        print(f"  Saved main visualization for K={k}")


def calculate_hourly_transitions(group, apps, all_states, transition_cols):
    """
    Helper function to calculate the flattened transition matrix for a
    single patient-hour (the 'group').

    NEW LOGIC: Ignores session_id. A transition is only valid if the
    gap between the OPEN of one app and the OPEN of the next is <= 5 minutes.
    """
    threshold_minutes = 5.0

    # 1. Sort by open time (group is already just 'open' events for one patient-hour)
    group = group.sort_values('timestamp')

    # 2. Find the next app and its open time (ignoring session_id)
    group['from_app'] = group['app']
    group['to_app'] = group['app'].shift(-1)
    group['to_timestamp'] = group['timestamp'].shift(-1) # This is the OPEN time of the next app

    # 3. Calculate the gap (in minutes) from this open to the next open
    if 'to_timestamp' in group.columns and not group['to_timestamp'].isnull().all():
        group['gap_in_minutes'] = (group['to_timestamp'] - group['timestamp']).dt.total_seconds() / 60.0
    else:
        group['gap_in_minutes'] = np.nan

    # 4. Filter for valid transitions (within the 5-minute threshold)
    transitions_df = group[group['gap_in_minutes'] <= threshold_minutes].copy()

    # 5. Count transitions
    if transitions_df.empty:
        # Return a zero-filled Series
        return pd.Series(0, index=transition_cols)

    counts = transitions_df.groupby(['from_app', 'to_app']).size().unstack(fill_value=0)

    # 6. Reindex and flatten
    matrix = counts.reindex(index=apps, columns=all_states).fillna(0.0)
    flat_vector = matrix.values.flatten()
    return pd.Series(flat_vector, index=transition_cols)


def create_hourly_features(events_df):
    """
    Converts raw event log into a featurized DataFrame, one row per patient-hour.
    This version CORRECTLY creates rows for all hours, including those with zero activity.

    Features created:
    - duration_<app>: Total minutes spent in each app.
    - count_<app>: Total number of times each app was opened.
    - t_<app>_to_<state>: Flattened transition matrix (raw counts).
    """
    print(f"Running featurization... (Loaded {len(events_df)} events)")

    # 1. Load app lists from config
    try:
        apps = _get('apps')
        all_states = _get('all_states')
    except KeyError as e:
        print(f"Error: {e} not found in config/general.yaml.")
        print("Please add 'apps' and 'all_states' lists to your config.")
        return pd.DataFrame(), []

    # 2. Prepare Base Event Data
    events_df['timestamp'] = pd.to_datetime(events_df['timestamp'])
    events_df_sorted = events_df.sort_values(by=['patient_id', 'timestamp']) # Sort by patient, then time

    # Find 'next_timestamp' to calculate durations
    events_df_sorted['next_timestamp'] = events_df_sorted.groupby(['patient_id'])['timestamp'].shift(-1)

    open_events = events_df_sorted[events_df_sorted['event_type'] == 'open'].copy()
    if open_events.empty:
        print("Warning: No 'open' events found in data.")
        return pd.DataFrame(), []

    open_events['duration_min'] = (open_events['next_timestamp'] - open_events['timestamp']).dt.total_seconds() / 60.0
    open_events['hour_ts'] = open_events['timestamp'].dt.floor('H')

    # 3. Create a base DataFrame with ALL patient-hours using resample
    # This is the key fix. resample('H') creates all the in-between hours.
    print("Creating base patient-hour index...")
    base_df = open_events.set_index('timestamp').groupby('patient_id')['app'].resample('H').count().to_frame(name='_dummy_count')
    base_df = base_df.drop(columns=['_dummy_count']).reset_index()
    base_df = base_df.rename(columns={'patient_id': 'patient', 'timestamp': 'hour_ts'})

    if base_df.empty:
        print("Error: Base DataFrame is empty after resample. Check data timestamps.")
        return pd.DataFrame(), []

    # --- Feature 1: Time Spent (Duration) ---
    print("Calculating duration features...")
    duration_features = open_events.groupby(['patient_id', 'hour_ts', 'app'])['duration_min'].sum().unstack(fill_value=0)
    duration_features = duration_features.reindex(columns=apps).fillna(0.0) # Ensure all apps
    duration_features.columns = [f'duration_{app}' for app in duration_features.columns]
    duration_features = duration_features.reset_index().rename(columns={'patient_id': 'patient'})

    # --- Feature 2: Open Count ---
    print("Calculating count features...")
    count_features = open_events.groupby(['patient_id', 'hour_ts', 'app']).size().unstack(fill_value=0)
    count_features = count_features.reindex(columns=apps).fillna(0.0) # Ensure all apps
    count_features.columns = [f'count_{app}' for app in count_features.columns]
    count_features = count_features.reset_index().rename(columns={'patient_id': 'patient'})

    # --- Feature 3: Transition Matrix ---
    print("Calculating transition features...")
    transition_cols = [f't_{fr}_to_{to}' for fr in apps for to in all_states]

    # We apply this to the 'open_events' grouped by hour
    transition_features = open_events.groupby(['patient_id', 'hour_ts']).apply(
        calculate_hourly_transitions,
        apps=apps,
        all_states=all_states,
        transition_cols=transition_cols
    ).reset_index().rename(columns={'patient_id': 'patient'})

    # 4. Merge all features onto the (now correct) base DataFrame
    print("Merging features...")
    features = base_df.merge(duration_features, on=['patient', 'hour_ts'], how='left')
    features = features.merge(count_features, on=['patient', 'hour_ts'], how='left')
    features = features.merge(transition_features, on=['patient', 'hour_ts'], how='left')

    # Fill NaNs with 0 (for all the hours that had no activity)
    features = features.fillna(0)

    # 5. Add final required columns
    features['week'] = features['hour_ts'].dt.to_period('W')

    # 6. Define final feature list
    final_feature_cols = [col for col in features.columns if col.startswith('duration_') or col.startswith('count_') or col.startswith('t_')]

    print(f"Featurization complete. Created {len(features)} hourly vectors with {len(final_feature_cols)} features each.")

    # Return both the features and the list of feature names
    return features, final_feature_cols


# ---
# MAIN (MODIFIED)
# ---

if __name__ == "__main__":

    # --- 1. Setup Argument Parser ---
    parser = argparse.ArgumentParser(description="Run clustering analysis on generated event data.")
    parser.add_argument(
        "-f", "--file",
        help="Path to the generated events.csv file",
        default="event_data_cache_sessions_distinct_lambda_3.csv" # <-- Default file
    )
    parser.add_argument(
        "-v", "--viz",
        help="Visualization method (tsne, pca, umap)",
        default="tsne"
    )
    args = parser.parse_args()

    print(f"Starting analysis on file: {args.file}")

    # --- 2. Load Config and Data ---
    try:
        PERSONA_MAP = _get('persona_map')
    except KeyError as e:
        print(f"Error: {e}. Please ensure 'persona_map' is defined in 'config/general.yaml'.")
        exit(1)

    try:
        events_df = pd.read_csv(args.file)
    except FileNotFoundError:
        print(f"Error: File not found at {args.file}")
        exit(1)

    # --- 3. Featurization ---
    # This now returns the feature list automatically
    hourly_df, feature_cols = create_hourly_features(events_df)

    if hourly_df.empty:
        print("Featurization returned an empty DataFrame. Exiting.")
        exit(1)

    if not feature_cols:
        print("Featurization returned no features. Exiting.")
        exit(1)

    # --- 4. Define Features & Split Data ---

    # We no longer need to manually define feature_cols!
    print(f"Using {len(feature_cols)} features for clustering.")

    # Split data: Test on the most recent week, train on all prior weeks
    max_week = hourly_df['week'].max()
    test_week = max_week

    train_df = hourly_df[hourly_df['week'] < test_week].copy()
    test_df = hourly_df[hourly_df['week'] == test_week].copy()

    if train_df.empty or test_df.empty:
        print(f"Error: Train (rows={len(train_df)}) or Test (rows={len(test_df)}) set is empty.")
        print("This can happen if you only generated 1 week of data. You need at least 2 weeks.")
        exit(1)

    print(f"Data split: {len(train_df)} train rows (Before week {test_week}), {len(test_df)} test rows (Week {test_week})")

    # --- 5. Run Pipeline ---
    title_prefix = f"Analysis for {os.path.basename(args.file)}"

    run_analysis_pipeline(
        train_df,
        test_df,
        feature_cols,
        PERSONA_MAP,
        visualization_method=args.viz,
        title_prefix=title_prefix
    )

    print("\nAnalysis complete.")