import os
import matplotlib
import yaml
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.use('Agg')  # <--- Add this line to fix the error
# --- Configuration Loading ---
CONFIG_PATH = 'config/general.yaml'
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Missing config: {CONFIG_PATH}")

with open(CONFIG_PATH, 'r') as _f:
    _cfg = yaml.safe_load(_f) or {}

def _get(k):
    return _cfg.get(k)

# --- Helper: JSON to Numpy Matrix ---
def json_matrix_to_numpy(period_matrix_dict, all_states):
    """Converts config dict to a sorted numpy array."""
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

# --- Main Visualization Logic ---
def plot_all_theoretical_matrices():
    # 1. Get Paths & States
    # Try both keys just in case of the typo mentioned earlier
    paths = _get('persona_types_path')
    if not paths:
        print("Error: No persona paths found in general.yaml")
        return

    all_states = _get('all_states')
    if not all_states:
        print("Error: 'all_states' not found in config.")
        return

    # 2. Collect Matrices
    # Structure: [ (PersonaName, TimePeriod, Matrix), ... ]
    matrix_collection = []

    print(f"Loading {len(paths)} persona configurations...")

    for p_name, p_path in paths.items():
        p_path = p_path.strip()
        if not os.path.exists(p_path):
            print(f"  Warning: Skipping missing file {p_path}")
            continue

        try:
            with open(p_path, 'r') as f:
                data = json.load(f)

            # Extract Matrices from 'period_matrices'
            if 'period_matrices' in data:
                # Use the order defined in 'day_parts' if available, else dict keys
                # This keeps Morning -> Noon -> Night logical order if defined
                ordered_periods = [dp['name'] for dp in data.get('day_parts', [])]
                if not ordered_periods:
                    ordered_periods = list(data['period_matrices'].keys())

                for period in ordered_periods:
                    if period in data['period_matrices']:
                        raw_mat = data['period_matrices'][period]
                        np_mat = json_matrix_to_numpy(raw_mat, all_states)

                        # Get hours for Label (e.g., "9-17")
                        hours_label = ""
                        if 'schedule' in data and period in data['schedule']:
                            s, e = data['schedule'][period]
                            hours_label = f"[{s}:00-{e}:00]"

                        label = f"{p_name}\n{period} {hours_label}"
                        matrix_collection.append((label, np_mat))

        except Exception as e:
            print(f"  Error reading {p_name}: {e}")

    # 3. Plotting
    if not matrix_collection:
        print("No matrices found.")
        return

    n_plots = len(matrix_collection)
    cols = 4  # How many columns in the grid
    rows = (n_plots // cols) + (1 if n_plots % cols > 0 else 0)

    fig_width = 4 * cols
    fig_height = 3.5 * rows

    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    axes = axes.flatten() # easier to index

    print(f"Generating {n_plots} plots...")

    for i, (label, matrix) in enumerate(matrix_collection):
        ax = axes[i]
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            cbar=False,
            xticklabels=all_states,
            yticklabels=all_states,
            ax=ax,
            annot_kws={"size": 8}
        )
        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=8)

    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()

    out_dir = _get('persona_dir')
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "theoretical_matrices.png")

    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Done! Saved comparison grid to: {save_path}")

if __name__ == "__main__":
    plot_all_theoretical_matrices()