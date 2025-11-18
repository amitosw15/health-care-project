import os
import matplotlib
matplotlib.use('Agg')  # Must be before importing pyplot

import matplotlib.pyplot as plt
# ... rest of your imports
# --- FIX 1: WORKAROUND FOR CUDA ERROR ---
# This forces TensorFlow to use your CPU.
# Remove this line when your GPU drivers are fixed.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# --- END FIX 1 ---

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
from sklearn.mixture import GaussianMixture
# --- NEW: Imports for VAE ---
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError:
    print("Error: TensorFlow not installed. 'pip install tensorflow'.")
    print("VAE clustering will not be available.")
    tf = None
# --- End VAE Imports ---


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

# --- Configuration (Copied from data_generation.py) ---
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config', 'general.yaml')
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Missing configuration file: {CONFIG_PATH}.")

with open(CONFIG_PATH, 'r') as _f:
    _cfg = yaml.safe_load(_f) or {}

def _get(k):
    """Return configuration value for key `k`. Raise if missing."""
    if k in _cfg:
        return _cfg[k]
    raise KeyError(f"Configuration key '{k}' missing in {CONFIG_PATH}. Add it to the YAML.")

# ---
# NEW: VAE MODEL DEFINITION
# ---
if tf:
    class Sampling(layers.Layer):
        """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
        def call(self, inputs):
            z_mean, z_log_var = inputs
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def build_vae(input_dim, latent_dim=16, hidden_dim=64, beta=1):
        """Builds the VAE model components."""

        # --- Encoder ---
        encoder_inputs = keras.Input(shape=(input_dim,))
        h = layers.Dense(hidden_dim, activation="relu")(encoder_inputs)
        h = layers.Dense(hidden_dim // 2, activation="relu")(h)
        z_mean = layers.Dense(latent_dim, name="z_mean")(h)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(h)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

        # --- Decoder ---
        latent_inputs = keras.Input(shape=(latent_dim,))
        h_decoded = layers.Dense(hidden_dim // 2, activation="relu")(latent_inputs)
        h_decoded = layers.Dense(hidden_dim, activation="relu")(h_decoded)
        decoder_outputs = layers.Dense(input_dim)(h_decoded) # No activation, linear output
        decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

        # --- VAE Class ---
        class VAE(keras.Model):
            def __init__(self, encoder, decoder, beta = 1,  **kwargs):
                super(VAE, self).__init__(**kwargs)
                self.encoder = encoder
                self.decoder = decoder
                self.beta = beta
                self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
                self.recon_loss_tracker = keras.metrics.Mean(name="recon_loss")
                self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

            @property
            def metrics(self):
                return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker]

            def train_step(self, data):
                with tf.GradientTape() as tape:
                    z_mean, z_log_var, z = self.encoder(data)
                    reconstruction = self.decoder(z)

                    # --- FIX 2: TENSORFLOW BUG ---
                    # Use the MeanSquaredError CLASS as suggested by the error
                    mse_loss = keras.losses.MeanSquaredError()
                    recon_loss = mse_loss(data, reconstruction)
                    # --- END FIX 2 ---

                    # KL Divergence
                    kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
                    kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

                    total_loss = recon_loss + self.beta*kl_loss

                grads = tape.gradient(total_loss, self.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

                self.total_loss_tracker.update_state(total_loss)
                self.recon_loss_tracker.update_state(recon_loss)
                self.kl_loss_tracker.update_state(kl_loss)

                return {
                    "loss": self.total_loss_tracker.result(),
                    "recon_loss": self.recon_loss_tracker.result(),
                    "kl_loss": self.kl_loss_tracker.result(),
                }

        vae = VAE(encoder, decoder, beta=beta)
        vae.compile(optimizer=keras.optimizers.Adam())
        return vae, encoder, decoder



import json
import glob

def load_persona_configs(persona_map, config_dir='config/personas'):
    """
    Loads individual persona configuration files (JSON) to extract schedules.
    Returns a dict: {'Influencer': {'Active': [8, 0], 'Sleep': [0, 8]}, ...}
    """
    schedules = {}
    unique_personas = set(persona_map.values())

    print(f"Loading persona schedules from: {config_dir}...")

    for persona in unique_personas:
        # Assume file is named "PersonaName.json"
        file_path = os.path.join(config_dir, f"{persona}.json")

        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    # Extract the 'schedule' key as requested
                    if 'schedule' in data:
                        schedules[persona] = data['schedule']
                        print(f"  -> Loaded schedule for {persona}")
            except Exception as e:
                print(f"  -> Error loading {persona}: {e}")
        else:
            print(f"  -> Warning: Config file not found for {persona} ({file_path})")

    return schedules
# ---
# ANALYSIS PIPELINE (MODIFIED)
# --
import json
import os
# Ensure other imports (numpy, pandas, plt, etc.) are present at top of file

def run_analysis_pipeline(train_df, test_df, feature_cols, persona_map, visualization_method='tsne', title_prefix=""):
    print(f"\n--- Running Pipeline: {title_prefix} (Viz: {visualization_method.upper()}) ---")

    if tf is None:
        print("TensorFlow is not installed. VAE pipeline cannot run.")
        return

    # --- 1. Load Schedules using 'persona_types_path' from YAML ---
    persona_schedules = {}

    try:
        # We try to get the key. We check for both 'persona_types_path' (correct)
        # and 'personaa_types_path' (your current typo) just in case.
        try:
            config_paths = _get('persona_types_path')
        except KeyError:
            config_paths = _get('personaa_types_path')

        print("Loading persona schedules from configured paths...")

        for persona_name, file_path in config_paths.items():
            # Safety: Clean up path if needed (remove whitespace)
            file_path = file_path.strip()

            if not os.path.exists(file_path):
                print(f"  [Warning] File not found: {file_path} (for {persona_name})")
                continue

            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if 'schedule' in data:
                        persona_schedules[persona_name] = data['schedule']
                        print(f"  [OK] Loaded schedule for '{persona_name}'")
                    else:
                        print(f"  [Warning] No 'schedule' key in {file_path}")
            except Exception as e:
                print(f"  [Error] Failed to load {file_path}: {e}")

    except KeyError:
        print("Error: Could not find 'persona_types_path' in config/general.yaml.")
        return

    # --- 2. Data Prep & VAE ---
    train_df = train_df.copy()
    test_df = test_df.copy()

    if train_df.empty or test_df.empty: return

    keep_cols = feature_cols + ['patient', 'hour_ts', 'week']
    train_df_filtered = train_df[keep_cols].dropna(subset=feature_cols)
    test_df_filtered = test_df[keep_cols].dropna(subset=feature_cols)

    train_features_raw = train_df_filtered[feature_cols].values
    test_features_raw = test_df_filtered[feature_cols].values

    train_patient_ids = train_df_filtered['patient'].values
    test_patient_ids = test_df_filtered['patient'].values
    train_labels = pd.Series(train_patient_ids).map(persona_map).values
    test_labels = pd.Series(test_patient_ids).map(persona_map).values

    if train_features_raw.shape[0] == 0: return

    print("Applying StandardScaler...")
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features_raw)
    test_features_scaled = scaler.transform(test_features_raw)

    print("Training VAE...")
    input_dim = train_features_scaled.shape[1]
    vae, encoder, decoder = build_vae(input_dim, latent_dim=16, hidden_dim=64, beta=0.05)
    vae.fit(train_features_scaled, epochs=30, batch_size=32, verbose=0)

    train_features = encoder.predict(train_features_scaled)[0]
    test_features = encoder.predict(test_features_scaled)[0]

    # --- 3. Visualization Coords ---
    print(f"Calculating {visualization_method.upper()} coordinates...")
    if visualization_method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        train_features_2d = reducer.fit_transform(train_features)
        test_features_2d = reducer.transform(test_features)
    elif visualization_method == 'umap' and umap is not None:
        reducer = umap.UMAP(n_components=2, n_neighbors=15, random_state=42)
        train_features_2d = reducer.fit_transform(train_features)
        test_features_2d = reducer.transform(test_features)
    else: # tsne
        combined = np.vstack((train_features, test_features))
        reducer = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
        features_2d = reducer.fit_transform(combined)
        train_features_2d = features_2d[:len(train_features)]
        test_features_2d = features_2d[len(train_features):]

    # --- Shape Logic (Using Loaded Schedules) ---
    def get_dynamic_shape_label(row):
        persona = row['persona']
        h = row['hour_ts'].hour

        if persona in persona_schedules:
            for label, time_range in persona_schedules[persona].items():
                # time_range is [start, end]
                start, end = time_range[0], time_range[1]

                # Case 1: Normal day range (e.g., 9 to 17)
                if start < end:
                    if start <= h < end:
                        return f"{persona} ({label})"
                # Case 2: Crosses Midnight or Full Day (e.g. 22 to 6, or 8 to 0)
                else:
                    if end == 0: # Treat 0 as midnight/24
                         if h >= start: return f"{persona} ({label})"
                    else:
                         if h >= start or h < end: return f"{persona} ({label})"

        return f"{persona} (Other)"

    # --- 4. Loop K Values ---
    k_values = [5, 10, 15, 20, 25]

    for k in k_values:
        print(f"\n>>> Processing K={k} <<<")

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        train_cluster_labels = kmeans.fit_predict(train_features)

        # Metrics
        h_train = homogeneity_score(train_labels, train_cluster_labels)
        c_train = completeness_score(train_labels, train_cluster_labels)
        v_train = v_measure_score(train_labels, train_cluster_labels)

        try:
            test_cluster_labels = kmeans.predict(test_features)
            h_test = homogeneity_score(test_labels, test_cluster_labels)
            c_test = completeness_score(test_labels, test_cluster_labels)
            v_test = v_measure_score(test_labels, test_cluster_labels)
            test_metrics_txt = f"K={k} Test:\nH:{h_test:.2f} C:{c_test:.2f} V:{v_test:.2f}"
        except:
            test_cluster_labels = np.array([-1]*len(test_features))
            test_metrics_txt = f"K={k} Test: Failed"

        # Plot Dataframes
        plot_train_df = pd.DataFrame({
            'vis_comp_0': train_features_2d[:, 0], 'vis_comp_1': train_features_2d[:, 1],
            'cluster': train_cluster_labels, 'patient': train_patient_ids,
            'hour_ts': train_df_filtered['hour_ts'].values
        })
        plot_test_df = pd.DataFrame({
            'vis_comp_0': test_features_2d[:, 0], 'vis_comp_1': test_features_2d[:, 1],
            'cluster': test_cluster_labels, 'patient': test_patient_ids,
            'hour_ts': test_df_filtered['hour_ts'].values,
            'week': test_df_filtered['week'].values
        })

        plot_train_df['persona'] = plot_train_df['patient'].map(persona_map)
        plot_test_df['persona'] = plot_test_df['patient'].map(persona_map)

        # Apply Dynamic Label
        plot_train_df['style_label'] = plot_train_df.apply(get_dynamic_shape_label, axis=1)
        plot_test_df['style_label'] = plot_test_df.apply(get_dynamic_shape_label, axis=1)

        # Visualization
        fig, (plot_ax, text_ax) = plt.subplots(nrows=1, ncols=2, figsize=(26, 12), gridspec_kw={'width_ratios': [4, 1]})
        palette = sns.color_palette('viridis', n_colors=k)

        sns.scatterplot(data=plot_train_df, x='vis_comp_0', y='vis_comp_1',
                        hue='cluster', style='style_label',
                        s=60, ax=plot_ax, palette=palette, alpha=0.6)

        sns.scatterplot(data=plot_test_df[plot_test_df['cluster'] != -1],
                        x='vis_comp_0', y='vis_comp_1',
                        hue='cluster', style='style_label',
                        s=200, ax=plot_ax, palette=palette, edgecolor='black', linewidth=1.5, legend=False)

        # Ellipses
        vis_features_train = plot_train_df[['vis_comp_0', 'vis_comp_1']].values
        cluster_means_vis = {c: vis_features_train[plot_train_df['cluster'] == c].mean(axis=0) for c in range(k) if not plot_train_df[plot_train_df['cluster'] == c].empty}
        plot_train_df['distance_vis'] = [mahalanobis(vis_features_train[i], cluster_means_vis[c], pinv(np.cov(vis_features_train[plot_train_df['cluster'] == c], rowvar=False) + np.eye(2)*1e-6)) if c in cluster_means_vis and len(vis_features_train[plot_train_df['cluster'] == c]) > 1 else np.nan for i, c in enumerate(plot_train_df['cluster'])]
        percentile_boundaries_vis = {c: np.nanpercentile(plot_train_df[plot_train_df['cluster'] == c]['distance_vis'], np.arange(10, 101, 10)) for c in range(k) if not plot_train_df[plot_train_df['cluster'] == c].empty}

        def confidence_ellipse(ax, center, cov, scale, **kwargs):
            try:
                lambda_, v = np.linalg.eigh(cov)
                lambda_ = np.sqrt(np.maximum(lambda_, 0))
                ell = Ellipse(xy=center, width=lambda_[0]*scale*2, height=lambda_[1]*scale*2, angle=np.rad2deg(np.arctan2(*v[:,0][::-1])), **kwargs)
                ax.add_patch(ell)
            except: pass

        for c in range(k):
            if c in cluster_means_vis and c in percentile_boundaries_vis:
                cluster_points = vis_features_train[plot_train_df['cluster'] == c]
                if len(cluster_points) > 1:
                    cov_vis = np.cov(cluster_points, rowvar=False) + np.eye(2)*1e-6
                    for p_val in percentile_boundaries_vis[c]:
                        if not np.isnan(p_val): confidence_ellipse(plot_ax, cluster_means_vis[c], cov_vis, scale=p_val, edgecolor=palette[c], facecolor='none', lw=1, alpha=0.6)

        plot_ax.set_title(f"K={k} | {visualization_method.upper()} | Dynamic Schedule Shapes", fontsize=16)
        plot_ax.legend(title="Clusters & Personas", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize='10')

        text_ax.axis('off')
        summary_text = (f"--- K={k} Train ---\nH:{h_train:.3f} C:{c_train:.3f} V:{v_train:.3f}\n\n{test_metrics_txt}")
        text_ax.text(0.05, 0.9, summary_text, transform=text_ax.transAxes, fontsize=14, family='monospace')

        param_dir = _get('param_dir')
        os.makedirs(param_dir, exist_ok=True)
        filename = os.path.join(param_dir, f"cluster_viz_k{k}_{title_prefix.lower().replace(' ', '_')}.png")
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved Plot: {filename}")

        # CSV Summary
        cluster_means = {c: train_features[plot_train_df['cluster'] == c].mean(axis=0) for c in range(k) if not plot_train_df[plot_train_df['cluster'] == c].empty}
        cluster_inv_covs = {c: pinv(np.cov(train_features[plot_train_df['cluster'] == c], rowvar=False) + np.eye(train_features.shape[1]) * 1e-6) for c in range(k) if train_features[plot_train_df['cluster'] == c].shape[0] > train_features.shape[1]}

        def calculate_mahalanobis(data, means, inv_covs, clusters):
            distances = [mahalanobis(data[i], means[c], inv_covs[c]) if c in means and c in inv_covs else np.nan for i, c in enumerate(clusters)]
            s_dist = pd.Series(distances)
            return s_dist.fillna(s_dist.mean())

        plot_train_df['distance'] = calculate_mahalanobis(train_features, cluster_means, cluster_inv_covs, plot_train_df['cluster'].values)
        plot_test_df['distance'] = calculate_mahalanobis(test_features, cluster_means, cluster_inv_covs, plot_test_df['cluster'].values)
        percentile_boundaries = {c: np.nanpercentile(plot_train_df[plot_train_df['cluster'] == c]['distance'], np.arange(10, 101, 10)) for c in range(k) if not plot_train_df[plot_train_df['cluster'] == c].empty}

        test_week_vectors = []
        for (patient, week), df_pw in plot_test_df.groupby(['patient', 'week']):
            week_vector = {'patient': patient, 'week': week, 'persona': persona_map.get(patient, "Unknown")}
            for c in range(k):
                distances = df_pw[df_pw['cluster'] == c]['distance']
                boundaries = percentile_boundaries.get(c, np.zeros(10))
                for p in range(10): week_vector[f'cluster{c}_p{p}'] = np.sum(distances > boundaries[p])
            test_week_vectors.append(week_vector)

        summary_table = pd.DataFrame(test_week_vectors)
        summary_filename = os.path.join(param_dir, f"final_summary_k{k}_{title_prefix.lower().replace(' ', '_')}.csv")
        summary_table.to_csv(summary_filename, index=False)
        print(f"Saved CSV: {summary_filename}")



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
    events_df_sorted = events_df.sort_values(by=['patient_id', 'session_id', 'timestamp'])
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

    # We still .apply() only on open_events, as transitions only happen then
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
    features['hour'] = features['hour_ts'].dt.hour
    features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24.0)
    features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24.0)
    # 6. Define final feature list
    final_feature_cols = [col for col in features.columns if col.startswith('duration_') or col.startswith('count_') or col.startswith('t_')]
    final_feature_cols.extend(['hour_sin', 'hour_cos']) # <-- ADD THIS
    print(f"Featurization complete. Created {len(features)} hourly vectors with {len(final_feature_cols)} features each.")
    print(f"Feature columns: {final_feature_cols}")
    # 7. Save a sample of the featurized data for inspection
    sample_filename = os.path.join(_get('param_dir'), 'featurized_data_sample.csv')
    features.head(100).to_csv(sample_filename, index=False)
    print(f"Sample of featurized data saved into: {sample_filename}")
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
        default="event_data_cache_sessions_distinct_lambda_10.csv" # <-- FIX 5: Default File
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