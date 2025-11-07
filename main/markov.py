import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
from datetime import datetime, timedelta
from scipy.spatial.distance import mahalanobis
from scipy.linalg import pinv
from matplotlib.patches import Ellipse

# VAE Imports
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mse

# --- Configuration ---
REGENERATE_DATA = False
EVENT_CACHE_PATH = 'event_data_cache_hybrid_4weeks.csv'
HOURLY_TRAIN_CACHE_PATH = 'train_hourly_cache_4weeks.csv'
HOURLY_TEST_CACHE_PATH = 'test_hourly_cache_4weeks.csv'

HOURS_PER_DAY = 24
DAYS_PER_WEEK = 7
EVENTS_PER_HOUR = 50 # Base average number of events per hour

np.random.seed(42)

# --- 1. Data Generation: A More Complex Population of Personas ---

apps = ['News', 'Social', 'Game', 'Work', 'Utility']

# --- Base Persona Matrices ---
persona_9_to_5er = pd.DataFrame.from_dict({
    'News':    {'News': 0.1, 'Social': 0.2, 'Game': 0.1, 'Work': 0.4, 'Utility': 0.2},'Social':  {'News': 0.2, 'Social': 0.1, 'Game': 0.2, 'Work': 0.3, 'Utility': 0.2},
    'Game':    {'News': 0.1, 'Social': 0.3, 'Game': 0.1, 'Work': 0.4, 'Utility': 0.1}, 'Work':    {'News': 0.1, 'Social': 0.1, 'Game': 0.1, 'Work': 0.1, 'Utility': 0.6},
    'Utility': {'News': 0.1, 'Social': 0.1, 'Game': 0.1, 'Work': 0.6, 'Utility': 0.1}}, orient='index')
persona_night_owl = pd.DataFrame.from_dict({
    'News':    {'News': 0.1, 'Social': 0.5, 'Game': 0.3, 'Work': 0.05, 'Utility': 0.05}, 'Social':  {'News': 0.4, 'Social': 0.1, 'Game': 0.4, 'Work': 0.05, 'Utility': 0.05},
    'Game':    {'News': 0.2, 'Social': 0.6, 'Game': 0.1, 'Work': 0.05, 'Utility': 0.05},'Work':    {'News': 0.2, 'Social': 0.4, 'Game': 0.2, 'Work': 0.1, 'Utility': 0.1},
    'Utility': {'News': 0.2, 'Social': 0.4, 'Game': 0.2, 'Work': 0.1, 'Utility': 0.1}}, orient='index')
persona_influencer = pd.DataFrame.from_dict({
    'News':    {'News': 0.1, 'Social': 0.8, 'Game': 0.05, 'Work': 0.0, 'Utility': 0.05},'Social':  {'News': 0.8, 'Social': 0.1, 'Game': 0.05, 'Work': 0.0, 'Utility': 0.05},
    'Game':    {'News': 0.4, 'Social': 0.4, 'Game': 0.1, 'Work': 0.0, 'Utility': 0.1}, 'Work':    {'News': 0.4, 'Social': 0.4, 'Game': 0.1, 'Work': 0.0, 'Utility': 0.1},
    'Utility': {'News': 0.4, 'Social': 0.4, 'Game': 0.1, 'Work': 0.0, 'Utility': 0.1}}, orient='index')
persona_compulsive_checker = pd.DataFrame.from_dict({
    'News':    {'News': 0.1, 'Social': 0.85, 'Game': 0.0, 'Work': 0.0, 'Utility': 0.05},'Social':  {'News': 0.1, 'Social': 0.85, 'Game': 0.0, 'Work': 0.0, 'Utility': 0.05},
    'Game':    {'News': 0.0, 'Social': 0.9, 'Game': 0.1, 'Work': 0.0, 'Utility': 0.0},'Work':    {'News': 0.0, 'Social': 0.9, 'Game': 0.0, 'Work': 0.1, 'Utility': 0.0},
    'Utility': {'News': 0.0, 'Social': 0.9, 'Game': 0.0, 'Work': 0.0, 'Utility': 0.1}}, orient='index')

hybrid_work_social = (persona_9_to_5er + persona_influencer) / 2
hybrid_leisure_compulsive = (persona_night_owl + persona_compulsive_checker) / 2

persona_matrices = {
    '9-to-5er': persona_9_to_5er, 'Night Owl': persona_night_owl, 'Influencer': persona_influencer,
    'Compulsive Checker': persona_compulsive_checker,
    'Work-Social Hybrid': hybrid_work_social, 'Leisure-Compulsive Hybrid': hybrid_leisure_compulsive
}

def generate_full_event_stream(num_weeks, patient_ids, persona_map):
    """Generates a rich event stream for a mixed population with varied event rates."""
    all_events = []
    start_date = datetime(2025, 1, 1)
    print("Generating full event stream for mixed population...")
    for patient in patient_ids:
        persona = persona_map[patient]
        transition_matrix = persona_matrices[persona]

        base_event_rate_scale = 3600 / EVENTS_PER_HOUR
        if persona == 'Compulsive Checker':
            event_rate_scale = base_event_rate_scale * 0.5
        else:
            event_rate_scale = base_event_rate_scale

        current_app = np.random.choice(apps)

        current_time = start_date
        end_time = start_date + timedelta(days=num_weeks * DAYS_PER_WEEK)

        while current_time < end_time:
            week = (current_time - start_date).days // 7
            hour_ts = current_time.replace(minute=0, second=0, microsecond=0)

            next_app_probs = transition_matrix.loc[current_app].values
            next_app = np.random.choice(apps, p=next_app_probs)

            time_to_next_event = np.random.exponential(scale=event_rate_scale)
            duration = np.random.exponential(scale=60)

            open_time = current_time + timedelta(seconds=time_to_next_event)
            if open_time >= end_time: break
            close_time = open_time + timedelta(seconds=duration)

            all_events.append((patient, week, hour_ts, open_time, f"APP_OPEN_{current_app}", 0))
            all_events.append((patient, week, hour_ts, close_time, f"APP_CLOSE_{current_app}", duration))

            current_time = close_time
            current_app = next_app

    return pd.DataFrame(all_events, columns=['patient', 'week', 'hour_ts', 'timestamp', 'event', 'duration'])

# --- VAE Implementation (Subclassed Model) ---
class VAE(tf.keras.Model):
    def __init__(self, original_dim, latent_dim=10, intermediate_dim=16, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.original_dim = original_dim
        self.latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim

        # Encoder
        encoder_inputs = Input(shape=(original_dim,))
        h = Dense(intermediate_dim, activation='relu')(encoder_inputs)
        self.z_mean = Dense(latent_dim)(h)
        self.z_log_var = Dense(latent_dim)(h)
        self.encoder = Model(encoder_inputs, [self.z_mean, self.z_log_var], name='encoder')

        # Decoder
        latent_inputs = Input(shape=(latent_dim,))
        x = Dense(intermediate_dim, activation='relu')(latent_inputs)
        outputs = Dense(original_dim, activation='sigmoid')(x)
        self.decoder = Model(latent_inputs, outputs, name='decoder')

    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)

        # Sampling
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon

        reconstruction = self.decoder(z)

        # Add KL loss
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        )
        self.add_loss(kl_loss)

        return reconstruction

def create_hybrid_features(train_df_raw, test_df_raw, compression_method='vae', n_components=10):
    """Creates hybrid feature vectors, now supporting VAE."""
    print(f"\nCreating hybrid features (Compression: {compression_method.upper()})...")

    def process_subset(df):
        if df.empty:
            index_cols = ['patient', 'week', 'hour_ts']
            reg_cols = [f"{stat}_{app}" for stat in ['count', 'total_duration', 'avg_duration'] for app in apps]
            empty_reg = pd.DataFrame(columns=index_cols + reg_cols).set_index(index_cols)
            empty_trans, empty_meta = np.array([]), pd.DataFrame(columns=index_cols).set_index(index_cols)
            return empty_reg, empty_trans, empty_meta

        regular_features_list = []
        for (patient, week, hour_ts), hour_df in df.groupby(['patient', 'week', 'hour_ts']):
            feat = {'patient': patient, 'week': week, 'hour_ts': hour_ts}
            app_events = hour_df[hour_df['event'].str.startswith('APP_CLOSE')]
            for app in apps:
                app_close_events = app_events[app_events['event'] == f"APP_CLOSE_{app}"]
                count, total_dur = len(app_close_events), app_close_events["duration"].sum()
                feat.update({f"count_{app}": count, f"total_duration_{app}": total_dur, f"avg_duration_{app}": (total_dur / count if count > 0 else 0)})
            regular_features_list.append(feat)
        regular_features_df = pd.DataFrame(regular_features_list).set_index(['patient', 'week', 'hour_ts'])

        hourly_matrices, metadata_list = [], []
        for (patient, week, hour_ts), hour_df in df.groupby(['patient', 'week', 'hour_ts']):
            metadata_list.append({'patient': patient, 'week': week, 'hour_ts': hour_ts})
            sequence = [event.split('_')[-1] for event in hour_df['event'] if 'APP' in event]
            counts_matrix = pd.DataFrame(0, index=apps, columns=apps)
            for i in range(len(sequence) - 1): counts_matrix.loc[sequence[i], sequence[i+1]] += 1
            prob_matrix = counts_matrix.div(counts_matrix.sum(axis=1), axis=0).fillna(0)
            hourly_matrices.append(prob_matrix.values.flatten())

        return regular_features_df, np.array(hourly_matrices), pd.DataFrame(metadata_list).set_index(['patient', 'week', 'hour_ts'])

    train_reg, train_trans_flat, train_meta = process_subset(train_df_raw)
    test_reg, test_trans_flat, test_meta = process_subset(test_df_raw)

    if compression_method == 'vae':
        original_dim = train_trans_flat.shape[1]
        vae = VAE(original_dim, latent_dim=n_components)
        vae.compile(optimizer='adam', loss=mse)
        print("Training VAE...")
        vae.fit(train_trans_flat, train_trans_flat, epochs=50, batch_size=256, validation_split=0.1, verbose=0)

        encoder = vae.encoder
        train_trans_final = encoder.predict(train_trans_flat)[0]
        test_trans_final = encoder.predict(test_trans_flat)[0] if test_trans_flat.size > 0 else np.array([])
        feature_names = [f'VAE_{i}' for i in range(n_components)]
    else:
        if compression_method == 'pca':
            reducer = PCA(n_components=n_components, random_state=42).fit(train_trans_flat)
        elif compression_method == 'umap':
            reducer = umap.UMAP(n_components=n_components, random_state=42).fit(train_trans_flat)
        else: raise ValueError("Invalid compression method")
        train_trans_final = reducer.transform(train_trans_flat)
        test_trans_final = reducer.transform(test_trans_flat) if test_trans_flat.size > 0 else np.array([])
        feature_names = [f'{compression_method.upper()}_{i}' for i in range(n_components)]

    train_trans_df = pd.DataFrame(train_trans_final, columns=feature_names, index=train_meta.index)
    if test_trans_final.size > 0:
        test_trans_df = pd.DataFrame(test_trans_final, columns=feature_names, index=test_meta.index)
    else:
        test_trans_df = pd.DataFrame(columns=feature_names, index=test_meta.index)

    train_hybrid = pd.concat([train_reg, train_trans_df], axis=1).reset_index()
    test_hybrid = pd.concat([test_reg, test_trans_df], axis=1).reset_index()

    return train_hybrid, test_hybrid

def run_analysis_pipeline(train_df, test_df, feature_cols, persona_map, visualization_method='tsne', title_prefix=""):
    """Runs the full pipeline, now with corrected visualization logic."""
    print(f"\n--- Running Pipeline: {title_prefix} (Viz: {visualization_method.upper()}) ---")

    train_features = train_df[feature_cols].values
    test_features = test_df[feature_cols].values

    if test_features.shape[0] == 0:
        print("WARNING: Test features array is empty. Skipping analysis.")
        return

    n_clusters = 6
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    train_df['cluster'] = kmeans.fit_predict(train_features)
    test_df['cluster'] = kmeans.predict(test_features)

    print(f"Applying {visualization_method.upper()} for visualization...")

    # FIX: Correctly fit reducer on train data and transform both train and test
    if visualization_method in ['pca', 'umap']:
        if visualization_method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
        else: # umap
            reducer = umap.UMAP(n_components=2, random_state=42)

        train_features_2d = reducer.fit_transform(train_features)
        test_features_2d = reducer.transform(test_features)

    elif visualization_method == 'tsne':
        # NOTE: t-SNE doesn't have a separate transform method. We fit it on the combined data for visualization purposes only.
        # This does not affect the clustering or summary table results.
        combined_features = np.vstack((train_features, test_features))
        reducer = TSNE(n_components=2, random_state=42, perplexity=30, n_jobs=-1)
        features_2d = reducer.fit_transform(combined_features)
        train_features_2d = features_2d[:len(train_features)]
        test_features_2d = features_2d[len(train_features):]
    else:
        raise ValueError("Invalid visualization method")

    train_df['vis_comp_0'], train_df['vis_comp_1'] = train_features_2d[:, 0], train_features_2d[:, 1]
    test_df['vis_comp_0'], test_df['vis_comp_1'] = test_features_2d[:, 0], test_features_2d[:, 1]

    def confidence_ellipse(ax, center, cov, scale, **kwargs):
        lambda_, v = np.linalg.eigh(cov)
        lambda_ = np.sqrt(lambda_)
        ell = Ellipse(xy=center, width=lambda_[0]*scale*2, height=lambda_[1]*scale*2, angle=np.rad2deg(np.arctan2(*v[:,0][::-1])), **kwargs)
        ax.add_patch(ell)

    vis_features_train = train_df[['vis_comp_0', 'vis_comp_1']].values
    cluster_means_vis = {c: vis_features_train[train_df['cluster'] == c].mean(axis=0) for c in range(n_clusters)}

    fig, ax = plt.subplots(figsize=(14, 12))
    palette = sns.color_palette('viridis', n_colors=n_clusters)
    train_df['persona'] = train_df['patient'].map(persona_map)
    test_df['persona'] = test_df['patient'].map(persona_map)

    sns.scatterplot(data=train_df, x='vis_comp_0', y='vis_comp_1', hue='cluster', style='persona', s=50, ax=ax, palette=palette, alpha=0.5, legend=False)
    sns.scatterplot(data=test_df, x='vis_comp_0', y='vis_comp_1', hue='cluster', style='persona', s=200, ax=ax, palette=palette, edgecolor='black', linewidth=1.5)

    train_df['distance_vis'] = [mahalanobis(vis_features_train[i], cluster_means_vis[c], pinv(np.cov(vis_features_train[train_df['cluster'] == c], rowvar=False))) if c in cluster_means_vis and len(vis_features_train[train_df['cluster'] == c]) > 1 else np.nan for i, c in enumerate(train_df['cluster'])]
    percentile_boundaries_vis = {c: np.nanpercentile(train_df[train_df['cluster'] == c]['distance_vis'], np.arange(10, 101, 10)) for c in range(n_clusters) if not train_df[train_df['cluster'] == c].empty}

    for c in range(n_clusters):
        if c in cluster_means_vis and c in percentile_boundaries_vis:
            center = cluster_means_vis[c]
            if len(vis_features_train[train_df['cluster'] == c]) > 1:
                cov_vis = np.cov(vis_features_train[train_df['cluster'] == c], rowvar=False)
                for p_val in percentile_boundaries_vis[c]:
                    confidence_ellipse(ax, center, cov_vis, scale=p_val, edgecolor=palette[c], facecolor='none', lw=1, alpha=0.6)

    title = f'{title_prefix}\n(Visualization: {visualization_method.upper()})'
    filename = f"cluster_viz_{title_prefix.lower().replace(' ', '_').replace(':', '')}_viz_{visualization_method}.png"
    ax.set_title(title, fontsize=16); ax.set_xlabel(f'{visualization_method.upper()} Component 1'); ax.set_ylabel(f'{visualization_method.upper()} Component 2')
    plt.savefig(filename); print(f"Saved cluster visualization to '{filename}'"); plt.close(fig)

    print("\nCalculating Mahalanobis distances and percentile boundaries...")
    cluster_means = {c: train_features[train_df['cluster'] == c].mean(axis=0) for c in range(n_clusters)}
    cluster_inv_covs = {c: pinv(np.cov(train_features[train_df['cluster'] == c], rowvar=False)) for c in range(n_clusters) if train_features[train_df['cluster'] == c].shape[0] > 1}

    def calculate_mahalanobis(data, means, inv_covs, clusters):
        distances = [mahalanobis(data[i], means[c], inv_covs[c]) if c in means and c in inv_covs else np.nan for i in range(len(data))]
        return pd.Series(distances).fillna(pd.Series(distances).mean())

    train_df['distance'] = calculate_mahalanobis(train_features, cluster_means, cluster_inv_covs, train_df['cluster'].values)
    test_df['distance'] = calculate_mahalanobis(test_features, cluster_means, cluster_inv_covs, test_df['cluster'].values)

    percentile_boundaries = {c: np.nanpercentile(train_df[train_df['cluster'] == c]['distance'], np.arange(10, 101, 10)) for c in range(n_clusters) if not train_df[train_df['cluster'] == c].empty}

    print("Building final summary table for test data...")
    test_week_vectors = []
    for (patient, week), df_pw in test_df.groupby(['patient', 'week']):
        week_vector = {'patient': patient, 'week': week, 'persona': persona_map[patient]}
        for c in range(n_clusters):
            distances = df_pw[df_pw['cluster'] == c]['distance']
            boundaries = percentile_boundaries.get(c, np.zeros(10))
            for p in range(10): week_vector[f'cluster{c}_p{p}'] = np.sum(distances > boundaries[p])
        test_week_vectors.append(week_vector)

    summary_table = pd.DataFrame(test_week_vectors)
    summary_filename = f"final_summary_{title_prefix.lower().replace(' ', '_').replace(':', '')}_viz_{visualization_method}.csv"
    summary_table.to_csv(summary_filename, index=False)
    print(f"Final Summary Table for Test Data:\n{summary_table}")
    print(f"Saved final summary table to '{summary_filename}'")


# --- Main Execution Block (Refactored for clarity) ---
persona_map = {
    0: '9-to-5er', 1: '9-to-5er', 2: 'Night Owl', 3: 'Night Owl',
    4: 'Influencer', 5: 'Influencer', 6: 'Work-Social Hybrid', 7: 'Work-Social Hybrid',
    8: 'Leisure-Compulsive Hybrid', 9: 'Leisure-Compulsive Hybrid',
    10: 'Compulsive Checker', 11: 'Compulsive Checker'
}
all_patient_ids = range(12)

# --- Step 1: Manage Raw Event Data ---
if REGENERATE_DATA or not os.path.exists(EVENT_CACHE_PATH):
    if not REGENERATE_DATA: print("Cached data not found. Forcing regeneration.")
    event_df = generate_full_event_stream(4, all_patient_ids, persona_map)
    event_df.to_csv(EVENT_CACHE_PATH, index=False)
else:
    print(f"Loading event data from cache: {EVENT_CACHE_PATH}")
    event_df = pd.read_csv(EVENT_CACHE_PATH)
    event_df['timestamp'] = pd.to_datetime(event_df['timestamp'])
    event_df['hour_ts'] = pd.to_datetime(event_df['hour_ts'])

# --- Step 2: Manage Hourly Feature Data ---
if REGENERATE_DATA or not os.path.exists(HOURLY_TRAIN_CACHE_PATH) or not os.path.exists(HOURLY_TEST_CACHE_PATH):
    print("\n--- Regenerating and Caching Hourly Features ---")

    print("\n--- Splitting data by week (Weeks 0-2 for Train, Week 3 for Test) ---")
    train_event_df = event_df[event_df['week'] < 3]
    test_event_df = event_df[event_df['week'] == 3]

    train_hourly_df, test_hourly_df = create_hybrid_features(
        train_event_df, test_event_df, compression_method='vae', n_components=10
    )
    train_hourly_df.to_csv(HOURLY_TRAIN_CACHE_PATH, index=False)
    test_hourly_df.to_csv(HOURLY_TEST_CACHE_PATH, index=False)
else:
    print("\n--- Loading Hourly Features from Cache ---")
    train_hourly_df = pd.read_csv(HOURLY_TRAIN_CACHE_PATH)
    test_hourly_df = pd.read_csv(HOURLY_TEST_CACHE_PATH)

# --- Step 3: Run Final Analysis ---
title_prefix = f"Hybrid Features (Matrix Comp: VAE)"
feature_cols = [col for col in train_hourly_df.columns if col not in ['patient', 'week', 'hour_ts']]

run_analysis_pipeline(train_hourly_df.copy(), test_hourly_df.copy(), feature_cols, persona_map,
                    visualization_method='tsne', title_prefix=title_prefix)

