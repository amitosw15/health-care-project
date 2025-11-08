import os
# --- NEW: SOLUTION 2 (IMMEDIATE WORKAROUND FOR CUDA ERROR) ---
# This forces TensorFlow to use your CPU.
# Remove this line when your GPU drivers are fixed.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# --- END NEW ---

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

    def build_vae(input_dim, latent_dim=16, hidden_dim=64):
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
            def __init__(self, encoder, decoder, **kwargs):
                super(VAE, self).__init__(**kwargs)
                self.encoder = encoder
                self.decoder = decoder
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

# As suggested by the error log, we instantiate the class
                    mse_loss = keras.losses.MeanSquaredError()
                    # We let the loss object calculate the mean loss
                    recon_loss = mse_loss(data, reconstruction)
                    # KL Divergence
                    kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
                    kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

                    total_loss = recon_loss + kl_loss

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

        vae = VAE(encoder, decoder)
        vae.compile(optimizer=keras.optimizers.Adam())
        return vae, encoder, decoder

# ---
# ANALYSIS PIPELINE (MODIFIED)
# ---

def run_analysis_pipeline(train_df, test_df, feature_cols, persona_map, visualization_method='tsne', title_prefix=""):
    """
    Runs the full pipeline, NOW WITH VAE:
    1. StandardScaler
    2. VAE (for feature extraction)
    3. K-Means (on latent features)
    4. Visualization (on latent features)
    """
    print(f"\n--- Running Pipeline: {title_prefix} (Viz: {visualization_method.upper()}) ---")

    if tf is None:
        print("TensorFlow is not installed. VAE pipeline cannot run.")
        return

    # Make copies to prevent modifying the original dataframes
    train_df = train_df.copy()
    test_df = test_df.copy()

    # --- Check for empty data BEFORE filtering features ---
    if train_df.empty or test_df.empty:
        print("WARNING: Initial train or test dataframe is empty. Skipping analysis.")
        return

    # --- Check for empty data AFTER filtering features ---
    if feature_cols:
        # Select relevant columns, *including time columns* for later summary
        keep_cols = feature_cols + ['patient', 'hour_ts', 'week']
        train_df_filtered = train_df[keep_cols].dropna(subset=feature_cols)
        test_df_filtered = test_df[keep_cols].dropna(subset=feature_cols)

        train_features_raw = train_df_filtered[feature_cols].values
        test_features_raw = test_df_filtered[feature_cols].values

        # Get corresponding labels
        train_labels = train_df_filtered['patient'].values
        test_labels = test_df_filtered['patient'].values
    else:
        # Handle case with no features
        print("Warning: No features selected. Using dummy data for shape.")
        train_features_raw = np.zeros((len(train_df), 1))
        test_features_raw = np.zeros((len(test_df), 1))
        train_labels = train_df['patient'].values
        test_labels = test_df['patient'].values

        # Need to add dummy time cols for filtered dfs
        train_df_filtered = train_df.copy()
        test_df_filtered = test_df.copy()

    if test_features_raw.shape[0] == 0 or train_features_raw.shape[0] == 0 or train_features_raw.shape[1] == 0:
        print("WARNING: Train or Test features array is empty after dropna or feature selection. Skipping analysis.")
        return

    # --- 1. Apply StandardScaler ---
    print("Applying StandardScaler to features...")
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features_raw)
    test_features_scaled = scaler.transform(test_features_raw)

    # --- 2. VAE Feature Extraction ---
    print("Building and training VAE for feature extraction...")
    input_dim = train_features_scaled.shape[1]
    latent_dim = 16  # You can tune this
    hidden_dim = 64  # You can tune this

    vae, encoder, decoder = build_vae(input_dim, latent_dim, hidden_dim)

    # Train the VAE on the scaled training data
    vae.fit(train_features_scaled, epochs=50, batch_size=32, verbose=0) # verbose=0 for cleaner output
    print("VAE training complete.")

    # Use the ENCODER to get the latent space representation (z)
    # We use z_mean for the "stable" representation
    train_z_mean, _, _ = encoder.predict(train_features_scaled)
    test_z_mean, _, _ = encoder.predict(test_features_scaled)

    print(f"Features transformed to latent space: {input_dim} -> {latent_dim}")

    # *** CRITICAL: All subsequent steps use the VAE's latent features ***
    train_features = train_z_mean
    test_features = test_z_mean

    # --- 3. K-Means Clustering (on VAE latent space) ---
    n_clusters = 25
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

    # Fit on train latent features, predict train labels
    train_cluster_labels = kmeans.fit_predict(train_features)

    # --- Calculate Train Metrics ---
    h_train = homogeneity_score(train_labels, train_cluster_labels)
    c_train = completeness_score(train_labels, train_cluster_labels)
    v_train = v_measure_score(train_labels, train_cluster_labels)
    print("\n--- Clustering Evaluation Metrics (Train, on VAE Features) ---")
    print(f"Homogeneity: {h_train:.4f} (Goal: ~1.0. Are clusters 'pure'?)")
    print(f"Completeness: {c_train:.4f} (Expected: < 1.0. Are classes split?)")
    print(f"V-Measure: {v_train:.4f} (Harmonic mean)")

    train_metrics_str = (
        f"--- Clustering Evaluation (Train) ---\n"
        f"(Features: VAE Latent Space)\n"
        f"Homogeneity:  {h_train:.4f}\n"
        f"Completeness: {c_train:.4f}\n"
        f"V-Measure:    {v_train:.4f}\n\n"
        f"Homogeneity is the goal (~1.0).\n"
        f"Completeness is expected to be lower."
    )

    # --- Predict Test Clusters & Metrics ---
    try:
        test_cluster_labels = kmeans.predict(test_features)

        h_test = homogeneity_score(test_labels, test_cluster_labels)
        c_test = completeness_score(test_labels, test_cluster_labels)
        v_test = v_measure_score(test_labels, test_cluster_labels)
        print("\n--- Clustering Evaluation Metrics (Test, on VAE Features) ---")
        print(f"Homogeneity: {h_test:.4f}")
        print(f"Completeness: {c_test:.4f}")
        print(f"V-Measure: {v_test:.4f}")

        test_metrics_str = (
            f"--- Clustering Evaluation (Test) ---\n"
            f"(Features: VAE Latent Space)\n"
            f"Homogeneity:  {h_test:.4f}\n"
            f"Completeness: {c_test:.4f}\n"
            f"V-Measure:    {v_test:.4f}"
        )

    except Exception as e:
        print(f"Error predicting test clusters: {e}. Test set might be too small or different.")
        test_cluster_labels = np.array([-1] * len(test_features)) # Assign dummy labels
        print("Skipping test set clustering metrics due to prediction error.")
        test_metrics_str = "--- Clustering Evaluation (Test) ---\n(Prediction Failed)"


    print(f"Applying {visualization_method.upper()} for visualization (on VAE latent space)...")

    # --- 4. Dimensionality Reduction (on VAE latent space) ---
    if train_features.shape[1] < 2:
        print("Warning: Fewer than 2 latent features. Visualizing raw latent features.")
        if train_features.shape[1] == 1:
            train_features_2d = np.hstack([train_features, np.zeros_like(train_features)])
            test_features_2d = np.hstack([test_features, np.zeros_like(test_features)])
        else:
             print("Error: No features to visualize.")
             return
    elif visualization_method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        train_features_2d = reducer.fit_transform(train_features)
        test_features_2d = reducer.transform(test_features)
    elif visualization_method == 'umap':
        if umap is None:
            print("Error: UMAP not installed. Falling back to PCA.")
            reducer = PCA(n_components=2, random_state=42)
            train_features_2d = reducer.fit_transform(train_features)
            test_features_2d = reducer.transform(test_features)
        else:
            # UMAP's n_neighbors needs to be less than the number of samples
            n_neighbors = min(15, train_features.shape[0] - 1)
            if n_neighbors <= 1:
                print("Warning: Not enough samples for UMAP. Falling back to PCA.")
                reducer = PCA(n_components=2, random_state=42)
                train_features_2d = reducer.fit_transform(train_features)
                test_features_2d = reducer.transform(test_features)
            else:
                reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
                train_features_2d = reducer.fit_transform(train_features)
                test_features_2d = reducer.transform(test_features)
    elif visualization_method == 'tsne':
        combined_features = np.vstack((train_features, test_features))
        perplexity_value = min(30, combined_features.shape[0] - 1)
        if perplexity_value <= 0:
             print("Warning: Not enough samples for t-SNE perplexity. Skipping visualization.")
             return
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity_value, n_jobs=-1, init='pca', learning_rate='auto')
        features_2d = reducer.fit_transform(combined_features)
        train_features_2d = features_2d[:len(train_features)]
        test_features_2d = features_2d[len(train_features):]
    else: raise ValueError("Invalid visualization method")

    # Ensure 2D for plotting
    if train_features_2d.shape[1] == 1:
        train_features_2d = np.hstack([train_features_2d, np.zeros_like(train_features_2d)])
        test_features_2d = np.hstack([test_features_2d, np.zeros_like(test_features_2d)])

    # --- (Plotting, Confidence Ellipse, and Summary Table code is UNCHANGED) ---
    # ... (it all correctly uses `train_features` which now points to `train_z_mean`) ...

    # --- Create Plotting DataFrames ---
    plot_train_df = pd.DataFrame({
        'vis_comp_0': train_features_2d[:, 0],
        'vis_comp_1': train_features_2d[:, 1],
        'cluster': train_cluster_labels,
        'patient': train_labels
    })
    plot_test_df = pd.DataFrame({
        'vis_comp_0': test_features_2d[:, 0],
        'vis_comp_1': test_features_2d[:, 1],
        'cluster': test_cluster_labels,
        'patient': test_labels
    })

    plot_train_df['persona'] = plot_train_df['patient'].map(persona_map)
    plot_test_df['persona'] = plot_test_df['patient'].map(persona_map)

    # --- BUG FIX ---
    # Add 'hour_ts' and 'week' back to the plot_dfs from the filtered_dfs
    plot_train_df['hour_ts'] = train_df_filtered['hour_ts'].values
    plot_train_df['week'] = train_df_filtered['week'].values
    plot_test_df['hour_ts'] = test_df_filtered['hour_ts'].values
    plot_test_df['week'] = test_df_filtered['week'].values
    # --- END BUG FIX ---


    def confidence_ellipse(ax, center, cov, scale, **kwargs):
      try:
        lambda_, v = np.linalg.eigh(cov)
        lambda_ = np.sqrt(np.maximum(lambda_, 0))
        ell = Ellipse(xy=center, width=lambda_[0]*scale*2, height=lambda_[1]*scale*2, angle=np.rad2deg(np.arctan2(*v[:,0][::-1])), **kwargs)
        ax.add_patch(ell)
      except np.linalg.LinAlgError:
        print(f"Warning: Could not compute ellipse for a cluster - likely singular covariance matrix.")


    vis_features_train = plot_train_df[['vis_comp_0', 'vis_comp_1']].values
    cluster_means_vis = {c: vis_features_train[plot_train_df['cluster'] == c].mean(axis=0) for c in range(n_clusters) if not plot_train_df[plot_train_df['cluster'] == c].empty}

    # Create Figure with 2 Subplots (Plot + Text)
    fig, (plot_ax, text_ax) = plt.subplots(
        nrows=1, ncols=2, # 1 row, 2 columns
        figsize=(20, 12),
        gridspec_kw={'width_ratios': [3, 1]} # 3:1 ratio for plot vs text
    )

    palette = sns.color_palette('viridis', n_colors=n_clusters)

    # Draw Scatter Plot on the left axes (`plot_ax`)
    sns.scatterplot(data=plot_train_df, x='vis_comp_0', y='vis_comp_1', hue='cluster', style='persona', s=50, ax=plot_ax, palette=palette, alpha=0.5, legend=False)
    sns.scatterplot(data=plot_test_df[plot_test_df['cluster'] != -1], x='vis_comp_0', y='vis_comp_1', hue='cluster', style='persona', s=200, ax=plot_ax, palette=palette, edgecolor='black', linewidth=1.5)

    plot_train_df['distance_vis'] = [mahalanobis(vis_features_train[i], cluster_means_vis[c], pinv(np.cov(vis_features_train[plot_train_df['cluster'] == c], rowvar=False) + np.eye(2)*1e-6)) if c in cluster_means_vis and len(vis_features_train[plot_train_df['cluster'] == c]) > 1 else np.nan for i, c in enumerate(plot_train_df['cluster'])]
    percentile_boundaries_vis = {c: np.nanpercentile(plot_train_df[plot_train_df['cluster'] == c]['distance_vis'], np.arange(10, 101, 10)) for c in range(n_clusters) if not plot_train_df[plot_train_df['cluster'] == c].empty}

    for c in range(n_clusters):
        if c in cluster_means_vis and c in percentile_boundaries_vis:
            center = cluster_means_vis[c]
            cluster_points = vis_features_train[plot_train_df['cluster'] == c]
            if len(cluster_points) > 1:
                cov_vis = np.cov(cluster_points, rowvar=False)
                cov_vis += np.eye(cov_vis.shape[0]) * 1e-6
                for p_val in percentile_boundaries_vis[c]:
                     if not np.isnan(p_val):
                        confidence_ellipse(plot_ax, center, cov_vis, scale=p_val, edgecolor=palette[c], facecolor='none', lw=1, alpha=0.6)

    title = f'{title_prefix}'

    # Save plot in the 'param_dir' from config
    param_dir = _get('param_dir')
    os.makedirs(param_dir, exist_ok=True)
    filename = os.path.join(param_dir, f"cluster_viz_{title_prefix.lower().replace(' ', '_').replace(':', '')}_viz_{visualization_method}.png")

    fig.suptitle(title, fontsize=20, y=1.03)
    plot_ax.set_title(f"Visualization: {visualization_method.upper()} (on VAE Latent Features)", fontsize=14)
    plot_ax.set_xlabel(f'{visualization_method.upper()} Component 1')
    plot_ax.set_ylabel(f'{visualization_method.upper()} Component 2')

    handles, labels = plot_ax.get_legend_handles_labels()
    unique_persona_handles = []
    unique_persona_labels = []
    seen_personas = set()
    for h, l in zip(reversed(handles), reversed(labels)):
        if l in persona_map.values() and l not in seen_personas:
            unique_persona_handles.insert(0, h)
            unique_persona_labels.insert(0, l)
            seen_personas.add(l)
    if unique_persona_handles:
        plot_ax.legend(unique_persona_handles, unique_persona_labels, title="Persona", loc='best')

    # Add Metrics Text to the right axes (`text_ax`)
    text_ax.axis('off')
    text_ax.text(0.0, 0.9, train_metrics_str,
                 transform=text_ax.transAxes,
                 fontsize=12, va='top', ha='left', family='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', ec='grey', lw=1, alpha=0.5))
    text_ax.text(0.0, 0.5, test_metrics_str, # Adjusted y-position
                 transform=text_ax.transAxes,
                 fontsize=12, va='top', ha='left', family='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', fc='ghostwhite', ec='grey', lw=1, alpha=0.5))

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight'); print(f"Saved cluster visualization to '{filename}'"); plt.close(fig)

    print("\nCalculating Mahalanobis distances (on VAE latent space)...")
    # Calculate Mahalanobis on VAE LATENT features
    cluster_means = {c: train_features[plot_train_df['cluster'] == c].mean(axis=0) for c in range(n_clusters) if not plot_train_df[plot_train_df['cluster'] == c].empty}
    cluster_inv_covs = {c: pinv(np.cov(train_features[plot_train_df['cluster'] == c], rowvar=False) + np.eye(train_features.shape[1]) * 1e-6) for c in range(n_clusters) if train_features[plot_train_df['cluster'] == c].shape[0] > train_features.shape[1]}

    def calculate_mahalanobis(data, means, inv_covs, clusters):
        distances = [mahalanobis(data[i], means[c], inv_covs[c]) if c in means and c in inv_covs else np.nan for i, c in enumerate(clusters)]
        s_dist = pd.Series(distances)
        mean_dist = s_dist.mean()
        return s_dist.fillna(mean_dist)

    plot_train_df['distance'] = calculate_mahalanobis(train_features, cluster_means, cluster_inv_covs, plot_train_df['cluster'].values) # VAE features
    plot_test_df['distance'] = calculate_mahalanobis(test_features, cluster_means, cluster_inv_covs, plot_test_df['cluster'].values)   # VAE features

    percentile_boundaries = {c: np.nanpercentile(plot_train_df[plot_train_df['cluster'] == c]['distance'], np.arange(10, 101, 10)) for c in range(n_clusters) if not plot_train_df[plot_train_df['cluster'] == c].empty}

    print("Building final summary table for test data...")
    test_week_vectors = []

    if 'week' not in plot_test_df.columns:
         print("Warning: 'week' column not found. Week-based summary may be inaccurate.")
         pass

    # Group by patient and week (now in plot_test_df)
    for (patient, week), df_pw in plot_test_df.groupby(['patient', 'week']):
        week_vector = {'patient': patient, 'week': week, 'persona': persona_map.get(patient, "Unknown")}
        for c in range(n_clusters):
            distances = df_pw[df_pw['cluster'] == c]['distance']
            boundaries = percentile_boundaries.get(c, np.zeros(10))
            for p in range(10): week_vector[f'cluster{c}_p{p}'] = np.sum(distances > boundaries[p])
        test_week_vectors.append(week_vector)

    summary_table = pd.DataFrame(test_week_vectors)
    summary_filename = os.path.join(param_dir, f"final_summary_{title_prefix.lower().replace(' ', '_').replace(':', '')}_viz_{visualization_method}.csv")
    summary_table.to_csv(summary_filename, index=False)
    print(f"Final Summary Table for Test Data:\n{summary_table}")
    print(f"Saved final summary table to '{summary_filename}'")


# ---
# FEATURIZATION (UNCHANGED)
# ---

def calculate_hourly_transitions(group, apps, all_states, transition_cols):
    """
    Helper function to calculate the flattened transition matrix for a
    single patient-hour (the 'group').
    """
    # 1. Find transitions within the hour
    group = group.sort_values('timestamp')
    group['from_app'] = group['app']
    group['to_app'] = group.groupby('session_id')['app'].shift(-1)

    transitions_df = group.dropna(subset=['to_app'])

    # 2. Count transitions
    if transitions_df.empty:
        # Return a zero-filled Series
        return pd.Series(0, index=transition_cols)

    counts = transitions_df.groupby(['from_app', 'to_app']).size().unstack(fill_value=0)

    # 3. Reindex to ensure all apps/states are present
    matrix = counts.reindex(index=apps, columns=all_states).fillna(0.0)

    # 4. Flatten and return as a Series (using raw counts)
    flat_vector = matrix.values.flatten()
    return pd.Series(flat_vector, index=transition_cols)


def create_hourly_features(events_df):
    """
    Converts raw event log into a featurized DataFrame, one row per patient-hour.
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

    # Find 'next_timestamp' to calculate durations
    events_df_sorted['next_timestamp'] = events_df_sorted.groupby(['patient_id', 'session_id'])['timestamp'].shift(-1)

    # Filter for 'open' events
    open_events = events_df_sorted[events_df_sorted['event_type'] == 'open'].copy()
    if open_events.empty:
        print("Warning: No 'open' events found in data.")
        return pd.DataFrame(), []

    # Calculate duration in minutes
    open_events['duration_min'] = (open_events['next_timestamp'] - open_events['timestamp']).dt.total_seconds() / 60.0

    # Assign the 'hour_ts' (the 00:00 of that hour)
    open_events['hour_ts'] = open_events['timestamp'].dt.floor('H')

    # 3. Create a base DataFrame of ALL possible patient-hours
    all_patients = events_df['patient_id'].unique()
    start_time = events_df['timestamp'].min().floor('H')
    end_time = events_df['timestamp'].max().floor('H')
    all_hours = pd.date_range(start_time, end_time, freq='H')

    base_index = pd.MultiIndex.from_product([all_patients, all_hours], names=['patient', 'hour_ts'])
    base_df = pd.DataFrame(index=base_index).reset_index()

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
    print("Calculating transition features (this may take a moment)...")
    # Define column names for the flattened vector
    transition_cols = [f't_{fr}_to_{to}' for fr in apps for to in all_states]

    # Group by patient and hour, then apply the helper function
    transition_features = open_events.groupby(['patient_id', 'hour_ts']).apply(
        calculate_hourly_transitions,
        apps=apps,
        all_states=all_states,
        transition_cols=transition_cols
    ).reset_index().rename(columns={'patient_id': 'patient'})

    # 4. Merge all features onto the base DataFrame
    print("Merging features...")
    features = base_df.merge(duration_features, on=['patient', 'hour_ts'], how='left')
    features = features.merge(count_features, on=['patient', 'hour_ts'], how='left')
    features = features.merge(transition_features, on=['patient', 'hour_ts'], how='left')

    # Fill NaNs created by the merge (for hours with 0 activity)
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
        default="event_data_cache_sessions_distinct_lambda_3.csv" # <-- CHANGED
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