import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json
import copy

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap
from datetime import datetime, timedelta
from scipy.spatial.distance import mahalanobis
from scipy.linalg import pinv
from matplotlib.patches import Ellipse
import random
import math

# VAE Imports
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mse

# Clustering Metrics
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score

# --- NEW: JSON Encoder for Numpy types ---
class NumpyAwareJSONEncoder(json.JSONEncoder):
    """
    A JSON encoder that can handle common numpy types (floats, ints, arrays)
    by converting them to standard Python types.
    """
    def default(self, obj):
        if isinstance(obj, (np.floating, np.float_)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int_)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# --- Configuration ---
# REGENERATE_DATA is now controlled by the main loop
EVENT_CACHE_PATH = 'event_data_cache_sessions_distinct.csv'
PARAM_DIR = 'persona_params' # store one params file per persona (persona_*.json)

HOURS_PER_DAY = 24
DAYS_PER_WEEK = 7
NUM_WEEKS = 4

np.random.seed(42)
random.seed(42)

# --- Base Data (unchanged) ---
apps = ['News', 'Social', 'Game', 'Work', 'Utility']
schedules = {
    '9-to-5er': {'Work': (9, 17), 'Recreation': (17, 23), 'Night': (23, 9)},
    'Night Owl': {'Sleep': (10, 18), 'Recreation': (18, 2), 'Morning': (2, 10)},
    'Influencer': {'Active': (8, 0), 'Sleep': (0, 8)},
    'Compulsive Checker': {'Work': (9, 17), 'Recreation': (17, 23), 'Night': (23, 9)},
}
p_9to5_work = pd.DataFrame({
    'News':    {'Work': 0.8, 'Utility': 0.15,'Social': 0.01, 'Game': 0.0, 'News': 0.04},
    'Social':  {'Work': 0.8, 'Utility': 0.15,'Social': 0.01, 'Game': 0.0, 'News': 0.04},
    'Game':    {'Work': 0.9, 'Utility': 0.05,'Social': 0.01, 'Game': 0.0, 'News': 0.04},
    'Work':    {'Work': 0.01,'Utility': 0.95,'Social': 0.01, 'Game': 0.0, 'News': 0.03},
    'Utility': {'Work': 0.95,'Utility': 0.01,'Social': 0.01, 'Game': 0.0, 'News': 0.03}},
    index=apps, columns=apps).fillna(0)
p_9to5_rec = pd.DataFrame({
    'News':    {'Social': 0.4, 'Game': 0.3, 'News': 0.1, 'Work': 0.05,'Utility': 0.15},
    'Social':  {'News': 0.3, 'Game': 0.3, 'Social': 0.1, 'Work': 0.05,'Utility': 0.25},
    'Game':    {'Social': 0.5, 'News': 0.2, 'Game': 0.1, 'Work': 0.05,'Utility': 0.15},
    'Work':    {'Social': 0.6, 'Game': 0.2, 'News': 0.1, 'Work': 0.0, 'Utility': 0.1},
    'Utility': {'Social': 0.6, 'Game': 0.2, 'News': 0.1, 'Work': 0.0, 'Utility': 0.1}},
    index=apps, columns=apps).fillna(0)
p_night_rec = pd.DataFrame({
    'News':    {'Social': 0.6, 'Game': 0.05,'News': 0.35,'Work': 0.0, 'Utility': 0.0},
    'Social':  {'News': 0.4, 'Game': 0.1, 'Social': 0.5, 'Work': 0.0, 'Utility': 0.0},
    'Game':    {'Social': 0.7, 'News': 0.1, 'Game': 0.8, 'Work': 0.0, 'Utility': 0.1},
    'Work':    {'Social': 0.9, 'Game': 0.0,'News': 0.1,'Work': 0.0, 'Utility': 0.0},
    'Utility': {'Social': 0.7, 'Game': 0.25,'News': 0.05,'Work': 0.0, 'Utility': 0.0}},
    index=apps, columns=apps).fillna(0)
p_influencer_active = pd.DataFrame({
    'News':    {'Social': 0.95,'News': 0.05,'Game': 0.0, 'Work': 0.0, 'Utility': 0.0},
    'Social':  {'News': 0.95,'Social': 0.05,'Game': 0.0, 'Work': 0.0, 'Utility': 0.0},
    'Game':    {'Social': 0.9, 'News': 0.1, 'Game': 0.0, 'Work': 0.0, 'Utility': 0.0},
    'Work':    {'Social': 0.9, 'News': 0.1, 'Game': 0.0, 'Work': 0.0, 'Utility': 0.0},
    'Utility': {'Social': 0.9, 'News': 0.1, 'Game': 0.0, 'Work': 0.0, 'Utility': 0.0}},
    index=apps, columns=apps).fillna(0)
p_compulsive_rec = pd.DataFrame({
    'News':    {'Social': 0.98,'News': 0.0,  'Game': 0.0, 'Work': 0.0, 'Utility': 0.02},
    'Social':  {'Social': 0.9, 'News': 0.05, 'Game': 0.0, 'Work': 0.0, 'Utility': 0.05},
    'Game':    {'Social': 1.0, 'Game': 0.0,  'News': 0.0, 'Work': 0.0, 'Utility': 0.0},
    'Work':    {'Social': 1.0, 'Work': 0.0,  'News': 0.0, 'Game': 0.0, 'Utility': 0.0},
    'Utility': {'Social': 1.0, 'Utility':0.0, 'News': 0.0, 'Game': 0.0, 'Work': 0.0}},
    index=apps, columns=apps).fillna(0)

# --- This is a *copy* of the original matrices, used for blending ---
original_base_matrices = {
    '9-to-5er': {'Work': p_9to5_work, 'Recreation': p_9to5_rec, 'Night': p_9to5_rec},
    'Night Owl': {'Sleep': p_9to5_rec, 'Recreation': p_night_rec, 'Morning': p_night_rec},
    'Influencer': {'Active': p_influencer_active, 'Sleep': p_9to5_rec},
    'Compulsive Checker': {'Work': p_9to5_work, 'Recreation': p_compulsive_rec, 'Night': p_compulsive_rec}
}
m_9to5_work = {'Work': 0.8, 'Utility': 0.15, 'News': 0.01, 'Social': 0.04, 'Game': 0.0}
m_9to5_rec = {'Social': 0.6, 'News': 0.1, 'Game': 0.1, 'Utility': 0.1, 'Work': 0.1}
m_compulsive_rec = {'Social': 0.9, 'News': 0.05, 'Utility': 0.05, 'Game': 0.0, 'Work': 0.0}
m_night_rec = {'Social': 0.7, 'Game': 0.25, 'News': 0.05, 'Utility': 0.0, 'Work': 0.0}
m_influencer_active = {'Social': 0.8, 'News': 0.15, 'Game': 0.0, 'Utility': 0.05, 'Work': 0.0}

base_period_marginals = {
    '9-to-5er': {'Work': m_9to5_work, 'Recreation': m_9to5_rec, 'Night': m_9to5_rec},
    'Night Owl': {'Sleep': m_9to5_rec, 'Recreation': m_night_rec, 'Morning': m_night_rec},
    'Influencer': {'Active': m_influencer_active, 'Sleep': m_9to5_rec},
    'Compulsive Checker': {'Work': m_9to5_work, 'Recreation': m_compulsive_rec, 'Night': m_compulsive_rec}
}
base_timing_params = {
    'session_start_dist': {'type': 'exponential', 'scale': 1800},
    'session_length_dist': {'type': 'poisson', 'lambda': 5}, # This value will be overridden by the loop
    'app_duration_dist': {'type': 'exponential', 'scale': 60},
    'inter_event_gap_dist': {'type': 'exponential', 'scale': 5}
}
compulsive_timing_params = copy.deepcopy(base_timing_params)
compulsive_timing_params['app_duration_dist']['scale'] = 30
# This will also be overridden
compulsive_timing_params['session_length_dist']['lambda'] = math.ceil(base_timing_params['session_length_dist']["lambda"] * 1.3)


# --- NEW: Function to blend persona matrices ---
def get_interpolated_matrices(alpha):
    """
    Blends persona matrices with the '9-to-5er' persona as the anchor.
    alpha = 0.0 means 100% unique (low similarity).
    alpha = 1.0 means 100% anchor (super high similarity).
    """
    # Anchor matrices (from 9-to-5er)
    anchor_work = original_base_matrices['9-to-5er']['Work']
    anchor_rec = original_base_matrices['9-to-5er']['Recreation']

    # 1. 9-to-5er (is the anchor, no change)
    matrices_9to5er = original_base_matrices['9-to-5er']

    # 2. Night Owl (blend all periods with anchor_rec)
    unique_sleep = original_base_matrices['Night Owl']['Sleep']
    unique_rec_night = original_base_matrices['Night Owl']['Recreation']
    unique_morning = original_base_matrices['Night Owl']['Morning']

    matrices_night_owl = {
        'Sleep': (alpha * anchor_rec) + ((1 - alpha) * unique_sleep),
        'Recreation': (alpha * anchor_rec) + ((1 - alpha) * unique_rec_night),
        'Morning': (alpha * anchor_rec) + ((1 - alpha) * unique_morning)
    }

    # 3. Influencer
    unique_active = original_base_matrices['Influencer']['Active']
    unique_sleep_inf = original_base_matrices['Influencer']['Sleep']

    matrices_influencer = {
        'Active': (alpha * anchor_rec) + ((1 - alpha) * unique_active),
        'Sleep': (alpha * anchor_rec) + ((1 - alpha) * unique_sleep_inf)
    }

    # 4. Compulsive Checker
    unique_work_comp = original_base_matrices['Compulsive Checker']['Work']
    unique_rec_comp = original_base_matrices['Compulsive Checker']['Recreation']
    unique_night_comp = original_base_matrices['Compulsive Checker']['Night']

    matrices_compulsive = {
        'Work': (alpha * anchor_work) + ((1 - alpha) * unique_work_comp),
        'Recreation': (alpha * anchor_rec) + ((1 - alpha) * unique_rec_comp),
        'Night': (alpha * anchor_rec) + ((1 - alpha) * unique_night_comp)
    }

    # Re-assemble the final dictionary
    final_matrices = {
        '9-to-5er': matrices_9to5er,
        'Night Owl': matrices_night_owl,
        'Influencer': matrices_influencer,
        'Compulsive Checker': matrices_compulsive
    }
    return final_matrices

def generate_and_save_persona_params(apps_list, directory, base_period_matrices):
    """
    Save one JSON per persona to `directory` using the provided base_period_matrices
    and the global base_period_marginals / timing params. Returns dict persona->params.
    JSON filename: persona_<slug>.json
    Each file contains:
      - persona, schedule, day_parts (list of 3 {name,start_hour,end_hour}),
      - period_matrices (dict-of-dicts), period_marginals (dict of dicts),
      - timing_params
    """
    print(f"Generating and saving persona parameters to '{directory}'...")
    os.makedirs(directory, exist_ok=True)
    persona_params = {}

    for persona, periods in base_period_matrices.items():
        # choose timing params
        timing = copy.deepcopy(compulsive_timing_params) if persona == 'Compulsive Checker' else copy.deepcopy(base_timing_params)

        # Build day_parts from schedules, convert (name,(start,e

