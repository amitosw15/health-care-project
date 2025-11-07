import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import mahalanobis
from scipy.linalg import pinv
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from matplotlib.lines import Line2D
import os # Add os for file path checking

# New imports for PCT
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.manifold import TSNE # Replaced UMAP with TSNE

# ------------------------
# Step 1. Generate Training and Test Data with Personas and Individual App Preferences
# ------------------------
np.random.seed(42)

# --- Configuration ---
REGENERATE_DATA = False # Set to True to re-run the data generation
TRAIN_DATA_FILE = 'train_hourly_df.csv'
TEST_DATA_FILE = 'test_hourly_df.csv'

all_apps = [f'app_{i}' for i in range(50)]
start_time = datetime(2025, 1, 1)
HOURS_PER_DAY = 24
DAYS_PER_WEEK = 7

# --- NEW: Generate persistent, sparse app preference profiles for each patient ---
def generate_patient_profiles(patient_ids, all_apps):
    """
    Generates a unique and persistent app preference profile for each patient.
    Each profile is a probability distribution over all available apps.
    """
    profiles = {}
    for patient in patient_ids:
        # Generate a base preference vector; most values will be low
        base_prefs = np.random.exponential(scale=1.0, size=len(all_apps))

        # Give each patient a few 'favorite' apps by boosting their scores
        num_favorites = np.random.randint(3, 8)
        favorite_indices = np.random.choice(len(all_apps), num_favorites, replace=False)
        base_prefs[favorite_indices] *= np.random.uniform(5, 15, size=num_favorites)

        # Normalize to get a probability distribution (softmax)
        profiles[patient] = np.exp(base_prefs) / np.sum(np.exp(base_prefs))
    return profiles

def generate_data(num_weeks, patient_ids, persona_map, patient_profiles):
    """
    Generates raw event data based on patient personas and individual app profiles.
    """
    events = []
    print(f"Generating {num_weeks} weeks of data for patients {patient_ids}...")
    for patient in patient_ids:
        persona = persona_map[patient]
        # Get this specific patient's fixed app probabilities
        app_probs = patient_profiles[patient]

        for day in range(num_weeks * DAYS_PER_WEEK):
            week = day // DAYS_PER_WEEK
            for hour in range(HOURS_PER_DAY):

                # Personas now only control activity level (n_events) and session duration
                n_events, duration_scale = (1, 30)

                if persona == '9-to-5er':
                    if 9 <= hour < 17: n_events, duration_scale = (2, 45)
                    elif 0 <= hour < 7: n_events, duration_scale = (0, 0)
                    else: n_events, duration_scale = (8, 75)

                elif persona == 'Night Owl':
                    if 17 <= hour < 23: n_events, duration_scale = (2, 45)
                    elif 8 <= hour < 16: n_events, duration_scale = (0, 0)
                    else: n_events, duration_scale = (10, 80)

                elif persona == 'Influencer':
                    n_events, duration_scale = (15, 90)

                elif persona == 'Compulsive Checker':
                    # Sporadic high-frequency, short-duration events
                    if np.random.rand() < 0.2:
                        n_events, duration_scale = (35, 5)
                    # Baseline 9-to-5er behavior
                    else:
                        if 9 <= hour < 17: n_events, duration_scale = (2, 45)
                        elif 0 <= hour < 7: n_events, duration_scale = (0, 0)
                        else: n_events, duration_scale = (8, 75)

                if n_events > 0:
                    for _ in range(n_events):
                        # The app is chosen from the patient's personal probability distribution
                        app = np.random.choice(all_apps, p=app_probs)
                        ts = start_time + timedelta(days=day, hours=hour, minutes=np.random.randint(0, 60), seconds=np.random.randint(0, 60))
                        duration = np.random.exponential(scale=duration_scale) if duration_scale > 0 else 0
                        events.append((patient, week, ts, app, duration))

    return pd.DataFrame(events, columns=["patient", "week", "timestamp", "app", "duration"])

# ------------------------
# Step 2. Create hourly feature vectors for both datasets
# ------------------------
def create_hourly_features(raw_df, apps_list):
    """
    Aggregates raw event data into hourly feature vectors.
    """
    raw_df["hour_ts"] = raw_df["timestamp"].dt.floor("H")
    features = []
    print("Creating hourly feature vectors...")
    for (patient, week, hour_ts), df_h in raw_df.groupby(["patient", "week", "hour_ts"]):
        feat = {"patient": patient, "week": week, "hour_ts": hour_ts}

        # Feature for compulsive behavior: count of short sessions
        feat['short_session_count'] = (df_h['duration'] < 60).sum()

        # Feature for compulsive behavior: minimum time between events
        if len(df_h) < 2:
            feat['min_time_between_events'] = 3600 # Default to 1 hour if not enough events
        else:
            sorted_events = df_h.sort_values('timestamp')
            time_diffs = sorted_events['timestamp'].diff().dt.total_seconds().dropna()
            feat['min_time_between_events'] = time_diffs.min() if not time_diffs.empty else 3600

        # App-specific features
        for app in apps_list: # Use the full list to ensure sparse vectors
            app_df = df_h[df_h["app"] == app]
            count, total_dur = len(app_df), app_df["duration"].sum()
            feat[f"count_{app}"], feat[f"total_duration_{app}"], feat[f"avg_duration_{app}"] = count, total_dur, (total_dur / count if count > 0 else 0)
        features.append(feat)
    return pd.DataFrame(features).fillna(0)

# --- Main data generation and loading flow ---
persona_map = {
    0: '9-to-5er', 1: '9-to-5er',
    2: 'Night Owl', 3: 'Night Owl',
    4: 'Influencer',
    5: 'Compulsive Checker', 6: 'Compulsive Checker'
}

if REGENERATE_DATA or not os.path.exists(TRAIN_DATA_FILE) or not os.path.exists(TEST_DATA_FILE):
    if not REGENERATE_DATA:
        print("Cached data not found. Forcing regeneration.")
    print("--- Running Data Generation and Feature Engineering ---")

    # Generate profiles for ALL patients first
    all_patient_ids = range(7)
    patient_profiles = generate_patient_profiles(all_patient_ids, all_apps)

    # Generate 10 weeks of training data
    train_patient_ids = range(7)
    train_raw_df = generate_data(10, train_patient_ids, persona_map, patient_profiles)

    # Generate 1 week of new "test" data for specific personas
    test_patient_ids = [0, 2, 4, 5]
    test_raw_df = generate_data(1, test_patient_ids, persona_map, patient_profiles)
    test_raw_df['week'] = 10 # Assign a new week number to avoid overlap

    # Create hourly feature vectors for both datasets
    train_hourly_df = create_hourly_features(train_raw_df, all_apps)
    test_hourly_df = create_hourly_features(test_raw_df, all_apps)

    # Save the processed dataframes to files
    print(f"Saving generated data to {TRAIN_DATA_FILE} and {TEST_DATA_FILE}...")
    train_hourly_df.to_csv(TRAIN_DATA_FILE, index=False)
    test_hourly_df.to_csv(TEST_DATA_FILE, index=False)

else:
    print(f"--- Loading Data from Cache: {TRAIN_DATA_FILE}, {TEST_DATA_FILE} ---")
    train_hourly_df = pd.read_csv(TRAIN_DATA_FILE)
    test_hourly_df = pd.read_csv(TEST_DATA_FILE)

# --- Define two sets of features for comparison ---
# 1. Full features including the new ones for compulsive behavior
app_feature_cols_full = [col for col in train_hourly_df.columns if 'count_app' in col or 'duration_app' in col]
app_feature_cols_full.append('short_session_count')
app_feature_cols_full.append('min_time_between_events')

# 2. Base features without the new ones (and without app counts)
app_feature_cols_base = [col for col in train_hourly_df.columns if 'duration_app' in col]


# ------------------------
# Steps 3-5. Encapsulated Pipeline Function
# ------------------------
def run_pipeline(train_df, test_df, feature_cols, file_suffix):
    """
    Runs the full PCT, clustering, visualization, and summary table generation pipeline.
    """
    print(f"\n{'='*20} RUNNING PIPELINE {file_suffix.replace('_', ' ').upper()} {'='*20}")

    # ------------------------
    # Step 3. Pre-Clustering Transformation (PCT) and Clustering
    # ------------------------
    def pre_clustering_transformation(train_data, test_data):
        """
        Applies a transformation pipeline (Scaler -> Autoencoder -> t-SNE)
        to the hourly feature data before clustering.
        """
        print("Applying Pre-Clustering Transformation (PCT)...")

        # 1. Scale data
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_data)
        test_scaled = scaler.transform(test_data)

        # 2. Autoencoder for dimensionality reduction
        input_dim = train_scaled.shape[1]
        encoding_dim = 16
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(64, activation='relu')(input_layer)
        encoded = Dense(32, activation='relu')(encoded)
        encoded = Dense(encoding_dim, activation='relu')(encoded)
        decoded = Dense(32, activation='relu')(encoded)
        decoded = Dense(64, activation='relu')(decoded)
        decoded = Dense(input_dim, activation='sigmoid')(decoded)

        autoencoder = Model(input_layer, decoded)
        encoder = Model(input_layer, encoded)

        autoencoder.compile(optimizer='adam', loss='mse')
        print("Training Autoencoder...")
        autoencoder.fit(train_scaled, train_scaled, epochs=20, batch_size=256, shuffle=True, verbose=0)

        train_latent = encoder.predict(train_scaled)
        test_latent = encoder.predict(test_scaled)

        # 3. t-SNE for final 2D embedding
        combined_latent = np.vstack((train_latent, test_latent))

        print("Applying t-SNE...")
        tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42, n_jobs=-1)
        combined_transformed = tsne.fit_transform(combined_latent)

        train_transformed = combined_transformed[:len(train_latent)]
        test_transformed = combined_transformed[len(train_latent):]

        return train_transformed, test_transformed, tsne

    train_features_for_clustering = train_df[feature_cols]
    test_features_for_clustering = test_df[feature_cols]

    train_transformed, test_transformed, tsne_reducer = pre_clustering_transformation(
        train_features_for_clustering,
        test_features_for_clustering
    )

    # --- Clustering is now performed on the TRANSFORMED data ---
    print("\nStep 3.5: Clustering on PCT-transformed data...")
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    train_df['cluster'] = kmeans.fit_predict(train_transformed)

    # --- Calculate Cluster Properties in the TRANSFORMED space ---
    cluster_means_transformed = {}
    cluster_covs_transformed = {}
    cluster_inv_covs_transformed = {}
    for c in range(n_clusters):
        cluster_data = train_transformed[train_df['cluster'] == c]
        cluster_means_transformed[c] = cluster_data.mean(axis=0)
        cov_matrix = np.cov(cluster_data, rowvar=False)
        cluster_covs_transformed[c] = cov_matrix
        cluster_inv_covs_transformed[c] = pinv(cov_matrix)

    # --- Calculate Mahalanobis Distance in TRANSFORMED space ---
    def calculate_mahalanobis_transformed(data, means, inv_covs, clusters):
        distances = []
        for i in range(len(data)):
            c = clusters[i]
            point = data[i]
            dist = mahalanobis(point, means[c], inv_covs[c])
            distances.append(dist)
        return distances

    train_df['distance_to_center'] = calculate_mahalanobis_transformed(train_transformed, cluster_means_transformed, cluster_inv_covs_transformed, train_df['cluster'].values)

    # --- Assign TEST data to existing clusters ---
    test_df['cluster'] = kmeans.predict(test_transformed)
    test_df['distance_to_center'] = calculate_mahalanobis_transformed(test_transformed, cluster_means_transformed, cluster_inv_covs_transformed, test_df['cluster'].values)

    # ------------------------
    # Step 4. Visualization now uses the t-SNE output
    # ------------------------
    print(f"\nStep 4: Creating new plot of clusters and test data points{file_suffix}...")

    def confidence_ellipse(ax, center, cov, scale, **kwargs):
        """
        Draw a confidence ellipse for a cluster.
        """
        lambda_, v = np.linalg.eigh(cov)
        lambda_ = np.sqrt(lambda_)
        ell = Ellipse(xy=center, width=lambda_[0]*scale*2, height=lambda_[1]*scale*2, angle=np.rad2deg(np.arctan2(*v[:,0][::-1])), **kwargs)
        ax.add_patch(ell)

    fig, ax = plt.subplots(figsize=(15, 12))
    palette = sns.color_palette('viridis', n_colors=n_clusters)
    ax.scatter(train_transformed[:, 0], train_transformed[:, 1], c=[palette[c] for c in train_df['cluster']], alpha=0.1, label='Training Data Hours')

    # Calculate percentile boundaries for Mahalanobis distances
    percentile_boundaries = {}
    for c in range(n_clusters):
        cluster_distances = train_df[train_df['cluster'] == c]['distance_to_center']
        if not cluster_distances.empty:
            percentile_boundaries[c] = np.percentile(cluster_distances, np.arange(10, 101, 10))
            center_tsne = cluster_means_transformed[c]
            cov_in_tsne_space = cluster_covs_transformed[c]
            # Draw ellipses corresponding to percentile boundaries
            for p_val in percentile_boundaries[c]:
                confidence_ellipse(ax, center_tsne, cov_in_tsne_space, scale=p_val, edgecolor=palette[c], facecolor='none', lw=2, alpha=0.7)

    # Overlay test data
    test_df['tsne1'] = test_transformed[:, 0]
    test_df['tsne2'] = test_transformed[:, 1]
    test_df['persona'] = test_df['patient'].map(persona_map)
    test_hourly_sample_df = test_df.sample(frac=0.1, random_state=42) # Sample to avoid clutter

    sns.scatterplot(
        data=test_hourly_sample_df, x='tsne1', y='tsne2', hue='cluster', style='persona',
        s=250, ax=ax, palette=palette, edgecolor='black', linewidth=1.5
    )

    ax.set_title(f'Hourly Behavior Clusters (in t-SNE space) with Test Data{file_suffix}', fontsize=16)
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.legend(title='Legend')

    viz_filename = f"cluster_pct_tsne_visualization{file_suffix}.png"
    plt.savefig(viz_filename)
    print(f"Saved new visualization to '{viz_filename}'")

    # ------------------------
    # Step 5. Build and display summary table for TEST data
    # ------------------------
    print(f"\nStep 5: Building final summary table for the NEW test weeks{file_suffix}...")
    test_week_vectors = []
    for (patient, week), df_pw in test_df.groupby(['patient', 'week']):
        week_vector = {'patient': patient, 'week': week, 'persona': persona_map[patient]}
        for c in range(n_clusters):
            patient_week_cluster_distances = df_pw[df_pw['cluster'] == c]['distance_to_center']
            boundaries = percentile_boundaries.get(c, np.zeros(10))
            for p in range(10):
                boundary = boundaries[p]
                # Count how many hourly vectors exceed this percentile boundary
                count = np.sum(patient_week_cluster_distances > boundary)
                week_vector[f'cluster{c}_p{p}'] = count
        test_week_vectors.append(week_vector)

    test_summary_table = pd.DataFrame(test_week_vectors)
    print(f"Final Summary Table for Test Data{file_suffix}:")
    print(test_summary_table)

    summary_filename = f"final_summary_table{file_suffix}.csv"
    test_summary_table.to_csv(summary_filename, index=False)
    print(f"Saved final summary table to '{summary_filename}'")


# --- Run Both Pipelines for Comparison ---
# Run 1: With the new compulsive behavior features
run_pipeline(train_hourly_df.copy(), test_hourly_df.copy(), app_feature_cols_full, "_with_new_features")

# Run 2: Without the new features, using only base app data
run_pipeline(train_hourly_df.copy(), test_hourly_df.copy(), app_feature_cols_base, "_without_new_features")


