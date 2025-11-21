import json
import os
import numpy as np


def json_matrix_to_numpy(period_matrix_dict, all_states):
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
