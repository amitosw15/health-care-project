import os
import math
import copy
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import oracle

# Short wrappers for convenience
PARAM_DIR = oracle.PARAM_DIR
EVENT_CACHE = oracle.EVENT_CACHE_PATH
APPS = oracle.apps
PERSONA_MAP = oracle.persona_map
ALL_PATIENT_IDS = list(oracle.all_patient_ids)

# Experiment defaults (kept from previous workflow)
SESSION_LENGTH_LAMBDAS = [3, 5, 10]
SIMILARITY_LEVELS = [
    (0.0, "low"),
    (0.4, "medium"),
    (0.8, "high"),
    (1.0, "super_high")
]

os.makedirs(PARAM_DIR, exist_ok=True)


def plot_and_save_persona_heatmaps(name_tag, persona_params, apps_list, outdir):
    """
    For a single persona params dict (loaded or produced), produce one PNG showing each period's
    transition matrix heatmap and right-side marginal bar, and annotate x-tick with to->app marginal.
    name_tag: string used in filename (e.g. persona_9-to-5er)
    """
    os.makedirs(outdir, exist_ok=True)
    fname = os.path.join(outdir, f"{name_tag}_matrices.png")

    periods = list(persona_params['period_matrices'].keys())
    n = len(periods)
    if n == 0:
        return None

    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig_w = cols * 5
    fig_h = rows * 4

    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(rows, cols * 2, figure=fig, width_ratios=[3, 1] * cols, wspace=0.4, hspace=0.6)

    for idx, period in enumerate(periods):
        r = idx // cols
        c = idx % cols
        left_col = c * 2
        right_col = left_col + 1

        # load matrix (might be dict-of-dicts if loaded from JSON)
        mat = persona_params['period_matrices'][period]
        if isinstance(mat, dict):
            df = pd.DataFrame.from_dict(mat, orient='index').reindex(index=apps_list, columns=apps_list).fillna(0.0)
        else:
            df = mat.reindex(index=apps_list, columns=apps_list).fillna(0.0).copy()

        # normalize rows (safety)
        row_sums = df.sum(axis=1).replace(0, 1)
        df = df.div(row_sums, axis=0)

        # compute to->app marginal (column sums normalized)
        col_sums = df.sum(axis=0).reindex(apps_list).fillna(0.0).astype(float)
        total_cols = float(col_sums.sum())
        if total_cols > 0:
            marg_vals = (col_sums / total_cols).tolist()
        else:
            marg_vals = [0.0 for _ in apps_list]

        # heatmap
        ax_heat = fig.add_subplot(gs[r, left_col])
        sns.heatmap(df, ax=ax_heat, cmap='viridis', vmin=0.0, vmax=1.0, cbar=False, annot=df.size <= 100, fmt=".2f",
                    linewidths=0.3, linecolor='lightgray')
        xticks = [f"{a}\n{v:.2f}" for a, v in zip(apps_list, marg_vals)]
        ax_heat.set_xticklabels(xticks, rotation=45, ha='right')
        ax_heat.set_yticklabels(ax_heat.get_yticklabels(), rotation=0)
        ax_heat.set_title(f"{persona_params.get('persona','') } — {period}")

        # marginal bar
        ax_bar = fig.add_subplot(gs[r, right_col])
        y_pos = list(range(len(apps_list)))
        bars = ax_bar.barh(y_pos, marg_vals, color='gray', edgecolor='black')
        ax_bar.set_xlim(0, max(0.01, max(marg_vals) * 1.1))
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(apps_list if c == 0 else [''] * len(apps_list))
        ax_bar.invert_yaxis()
        ax_bar.set_xlabel("Marginal")
        ax_bar.xaxis.set_label_position('top')
        # annotate numbers
        max_m = max(marg_vals) if marg_vals else 0
        offset = 0.005 if max_m <= 0.2 else max_m * 0.02
        for i, v in enumerate(marg_vals):
            ax_bar.text(v + offset, i, f"{v:.2f}", va='center', fontsize=8, color='black')

    plt.tight_layout()
    plt.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    return fname


def run_one_experiment(session_lambda, similarity_alpha, similarity_tag):
    # set timing params on oracle module (oracle functions read these globals)
    oracle.base_timing_params['session_length_dist']['lambda'] = session_lambda
    oracle.compulsive_timing_params = copy.deepcopy(oracle.base_timing_params)
    oracle.compulsive_timing_params['app_duration_dist']['scale'] = 30
    oracle.compulsive_timing_params['session_length_dist']['lambda'] = math.ceil(
        oracle.base_timing_params['session_length_dist']['lambda'] * 1.3
    )

    # build blended matrices
    current_base_matrices = oracle.get_interpolated_matrices(similarity_alpha)

    # generate and save one JSON per persona
    persona_params = oracle.generate_and_save_persona_params(APPS, PARAM_DIR, current_base_matrices)

    # map each patient to their persona params (no per-patient variation)
    theoretical_patient_params = {pid: copy.deepcopy(persona_params[PERSONA_MAP[pid]]) for pid in ALL_PATIENT_IDS}

    # generate events using per-patient (persona-shared) params
    event_df = oracle.generate_full_event_stream_sessions(oracle.NUM_WEEKS, ALL_PATIENT_IDS, PERSONA_MAP, theoretical_patient_params)
    event_df.to_csv(EVENT_CACHE, index=False)

    # Produce one image per persona (persona JSON already saved in PARAM_DIR)
    saved_images = []
    for persona_name, persona_params_dict in persona_params.items():
        name_tag = f"persona_{persona_name.replace(' ', '_')}"
        img_path = plot_and_save_persona_heatmaps(name_tag, persona_params_dict, APPS, PARAM_DIR)
        if img_path:
            saved_images.append(img_path)

    return event_df, saved_images


if __name__ == "__main__":
    results = []
    for session_lambda in SESSION_LENGTH_LAMBDAS:
        for alpha, tag in SIMILARITY_LEVELS:
            print(f"\n=== RUN: lambda={session_lambda}  alpha={alpha} ({tag}) ===")
            ev_df, imgs = run_one_experiment(session_lambda, alpha, tag)
            print(f"Saved events to {EVENT_CACHE} (rows={len(ev_df)})")
            print(f"Saved {len(imgs)} persona matrix images to {PARAM_DIR}")
            results.append({'lambda': session_lambda, 'alpha': alpha, 'images': imgs, 'events_rows': len(ev_df)})

    # Summarize
    summary_path = os.path.join(PARAM_DIR, "generation_summary.json")
    with open(summary_path, 'w') as sf:
        json.dump(results, sf, indent=2)
    print(f"\nGeneration complete. Summary saved to {summary_path}")