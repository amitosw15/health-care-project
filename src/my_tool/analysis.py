"""Analysis helpers that wrap the pipeline functions."""
import os
import typer

try:
    from ... import oracle as _oracle
except Exception:
    import oracle as _oracle


def run_analysis(experiment_name: str = None):
    """Run a default analysis: load data, create features, run pipeline for one experiment.

    This is intentionally small: it demonstrates wiring a CLI to the analysis functions.
    """
    if not os.path.exists(_oracle.EVENT_CACHE_PATH):
        typer.echo(f"Event cache '{_oracle.EVENT_CACHE_PATH}' not found. Generate data first.")
        return

    # Load params and event cache
    patient_ids = list(_oracle.all_patient_ids)
    params = _oracle.load_patient_params(patient_ids, _oracle.apps, _oracle.PARAM_DIR)
    df = _oracle.pd.read_csv(_oracle.EVENT_CACHE_PATH)
    df['timestamp'] = _oracle.pd.to_datetime(df['timestamp'])

    # Split weeks (train/test)
    train_event_df = df[df['week'] < (_oracle.NUM_WEEKS - 1)]
    test_event_df = df[df['week'] == (_oracle.NUM_WEEKS - 1)]

    train_hourly_df = _oracle.create_all_features(train_event_df, _oracle.persona_map, params, _oracle.apps)
    test_hourly_df = _oracle.create_all_features(test_event_df, _oracle.persona_map, params, _oracle.apps)

    # Run one experiment using real_prob only (as an example)
    real_prob_cols = [f'real_prob_{i}' for i in range(len(_oracle.apps) * len(_oracle.apps))]
    metrics = _oracle.run_analysis_pipeline(train_hourly_df, test_hourly_df, real_prob_cols, _oracle.persona_map, visualization_method='pca', title_prefix=(experiment_name or 'cli_run'))
    typer.echo(f"Analysis metrics: {metrics}")
