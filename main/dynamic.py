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

# --- Configuration ---
REGENERATE_DATA = False
EVENT_CACHE_PATH = 'event_data_cache_hybrid.csv'
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

# --- NEW: Create Hybrid Personas by mixing matrices ---
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

        # NEW: Compulsive checkers have a higher event rate (more events per hour)
        base_event_rate_scale = 3600 / EVENTS_PER_HOUR
        if persona == 'Compulsive Checker':
            event_rate_scale = base_event_rate_scale * 0.5 # Double the events
        else:
            event_rate_scale = base_event_rate_scale

        current_time = start_date
        current_app = np.random.choice(apps)

        for week in range(num_weeks):
            for day in range(DAYS_PER_WEEK):
                day_end_time = start_date + timedelta(days=week*DAYS_PER_WEEK + day + 1)
                while current_time < day_end_time:
                    hour_ts = current_time.replace(minute=0, second=0, microsecond=0)
                    next_app_probs = transition_matrix.loc[current_app].values
                    next_app = np.random.choice(apps, p=next_app_probs)
                    time_to_next_event = np.random.exponential(scale=event_rate_scale)
                    duration = np.random.exponential(scale=60)
                    open_time = current_time + timedelta(seconds=time_to_next_event)
                    if open_time >= day_end_time: break
                    close_time = open_time + timedelta(seconds=duration)
                    all_events.append((patient, week, hour_ts, open_time, f"APP_OPEN_{current_app}", 0))
                    all_events.append((patient, week, hour_ts, close_time, f"APP_CLOSE_{current_app}", duration))
                    current_time = close_time
                    current_app = next_app

    return pd.DataFrame(all_events, columns=['patient', 'week', 'hour_ts', 'timestamp', 'event', 'duration'])

# --- 2. Hybrid Feature Engineering (Unchanged) ---
def create_hybrid_features(df, compression_method='pca', n_components=10):
    """Creates a hybrid feature vector combining regular stats and compressed/flattened Markov models."""
    print(f"\nCreating hybrid features (Compression: {compression_method.upper()})...")
    # ... (Function body is unchanged) ...
    # Part A: Calculate regular features
    regular_features_list = []
    for (patient, week, hour_ts), hour_df in df.groupby(['patient', 'week', 'hour_ts']):
        feat = {'patient': patient, 'week': week, 'hour_ts': hour_ts}
        app_events = hour_df[hour_df['event'].str.startswith('APP_CLOSE')]
        for app in apps:
            app_close_events = app_events[app_events['event'] == f"APP_CLOSE_{app}"]
            count = len(app_close_events)
            total_dur = app_close_events["duration"].sum()
            feat.update({f"count_{app}": count, f"total_duration_{app}": total_dur, f"avg_duration_{app}": (total_dur / count if count > 0 else 0)})
        regular_features_list.append(feat)
    regular_features_df = pd.DataFrame(regular_features_list).set_index(['patient', 'week', 'hour_ts'])

    # Part B: Calculate and optionally compress transition matrices
    hourly_matrices, metadata_list = [], []
    for (patient, week, hour_ts), hour_df in df.groupby(['patient', 'week', 'hour_ts']):
        metadata_list.append({'patient': patient, 'week': week, 'hour_ts': hour_ts})
        sequence = [event.split('_')[-1] for event in hour_df['event'] if 'APP' in event]
        counts_matrix = pd.DataFrame(0, index=apps, columns=apps)
        for i in range(len(sequence) - 1):
            counts_matrix.loc[sequence[i], sequence[i+1]] += 1
        prob_matrix = counts_matrix.div(counts_matrix.sum(axis=1), axis=0).fillna(0)
        hourly_matrices.append(prob_matrix.values.flatten())

    if compression_method == 'pca':
        reducer = PCA(n_components=n_components, random_state=42)
        final_transition_features = reducer.fit_transform(hourly_matrices)
        feature_names = [f'PC_{i}' for i in range(n_components)]
    elif compression_method == 'umap':
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        final_transition_features = reducer.fit_transform(hourly_matrices)
        feature_names = [f'UMAP_{i}' for i in range(n_components)]
    elif compression_method == 'none':
        final_transition_features = np.array(hourly_matrices)
        feature_names = [f'flat_{i}' for i in range(final_transition_features.shape[1])]
    else: raise ValueError("Invalid compression method")

    transition_features_df = pd.DataFrame(final_transition_features, columns=feature_names)
    meta_df = pd.DataFrame(metadata_list).set_index(['patient', 'week', 'hour_ts'])
    transition_features_df.index = meta_df.index

    hybrid_df = pd.concat([regular_features_df, transition_features_df], axis=1)
    return hybrid_df.reset_index()

# --- 3. The Analysis Pipeline ---

def run_analysis_pipeline(df, feature_cols, persona_map, visualization_method='pca', title_prefix=""):
    """Runs clustering and visualization, allowing for different visualization methods."""
    print(f"\n--- Running Pipeline: {title_prefix} (Viz: {visualization_method.upper()}) ---")

    features = df[feature_cols].values

    print(f"Applying {visualization_method.upper()} for visualization...")
    if visualization_method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        features_2d = reducer.fit_transform(features)
    elif visualization_method == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=42)
        features_2d = reducer.fit_transform(features)
    elif visualization_method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30, n_jobs=-1)
        features_2d = reducer.fit_transform(features)
    else: raise ValueError("Invalid visualization method")

    df['vis_comp_0'] = features_2d[:, 0]
    df['vis_comp_1'] = features_2d[:, 1]

    # NEW: Increased number of clusters to find more nuanced patterns
    n_clusters = 6
    print(f"Clustering on full hybrid features into {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(features)

    print("Visualizing clusters...")
    fig, ax = plt.subplots(figsize=(14, 12))
    df['persona'] = df['patient'].map(persona_map)

    sns.scatterplot(data=df, x='vis_comp_0', y='vis_comp_1', hue='cluster', style='persona', s=100, ax=ax, palette='viridis')

    title = f'{title_prefix}\n(Visualization: {visualization_method.upper()})'
    filename = f"cluster_viz_{title_prefix.lower().replace(' ', '_').replace(':', '')}_viz_{visualization_method}.png"

    ax.set_title(title, fontsize=16)
    ax.set_xlabel(f'{visualization_method.upper()} Component 1')
    ax.set_ylabel(f'{visualization_method.upper()} Component 2')
    plt.savefig(filename)
    print(f"Saved cluster visualization to '{filename}'")
    plt.close(fig)

    cluster_persona_counts = df.groupby(['cluster', 'persona']).size().unstack(fill_value=0)
    print("\nCluster-Persona Purity Check:")
    print(cluster_persona_counts)

# --- Main Execution Block ---

# NEW: Expanded and mixed population
persona_map = {
    0: '9-to-5er', 1: '9-to-5er',
    2: 'Night Owl', 3: 'Night Owl',
    4: 'Influencer', 5: 'Influencer',
    6: 'Work-Social Hybrid', 7: 'Work-Social Hybrid',
    8: 'Leisure-Compulsive Hybrid', 9: 'Leisure-Compulsive Hybrid',
    10: 'Compulsive Checker', 11: 'Compulsive Checker' # Only 2 compulsive users
}
all_patient_ids = range(12)

if REGENERATE_DATA or not os.path.exists(EVENT_CACHE_PATH):
    if not REGENERATE_DATA: print("Cached data not found. Forcing regeneration.")
    event_df = generate_full_event_stream(10, all_patient_ids, persona_map)
    event_df.to_csv(EVENT_CACHE_PATH, index=False)
else:
    print(f"Loading event data from cache: {EVENT_CACHE_PATH}")
    event_df = pd.read_csv(EVENT_CACHE_PATH)

# --- Run Experiments (Unchanged) ---
compression_methods = ['pca', 'umap', 'none']
visualization_methods = ['pca', 'umap', 'tsne']

for comp_method in compression_methods:
    n_components = 10 if comp_method != 'none' else 25

    if comp_method == 'none':
        title_prefix = "Hybrid Features (No Matrix Compression)"
        hourly_feature_df = create_hybrid_features(event_df, compression_method='none')
    else:
        title_prefix = f"Hybrid Features (Matrix Comp: {comp_method.upper()})"
        hourly_feature_df = create_hybrid_features(event_df, compression_method=comp_method, n_components=n_components)

    feature_cols = [col for col in hourly_feature_df.columns if col not in ['patient', 'week', 'hour_ts']]

    for viz_method in visualization_methods:
         run_analysis_pipeline(hourly_feature_df.copy(), feature_cols, persona_map, visualization_method=viz_method, title_prefix=title_prefix)

