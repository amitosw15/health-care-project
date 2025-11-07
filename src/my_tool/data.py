"""Data helpers that wrap existing generation functions from the repo."""
import os
from typing import Iterable
import typer

# Import functions from the existing top-level `oracle.py`
try:
    from ... import oracle as _oracle  # relative import fallback (won't work when run from root)
except Exception:
    import oracle as _oracle


def _parse_patient_ids(patient_ids: Iterable[int]) -> list:
    return list(patient_ids)


def create_params(similarity_alpha: float = 1.0, out_dir: str = "patient_params"):
    """Generate theoretical patient parameters and save to JSON files.

    Very small wrapper around oracle.generate_and_save_patient_params.
    """
    # Use defaults defined in oracle module
    patient_ids = list(_oracle.all_patient_ids)
    persona_map = _oracle.persona_map
    apps_list = _oracle.apps
    base_period_matrices = _oracle.get_interpolated_matrices(similarity_alpha)

    os.makedirs(out_dir, exist_ok=True)
    _oracle.generate_and_save_patient_params(patient_ids, persona_map, apps_list, out_dir, base_period_matrices)


def generate_event_data(num_weeks: int = 4, out_csv: str = "event_data_cache_sessions_distinct.csv"):
    """Generate event stream using oracle.generate_full_event_stream_sessions and save CSV."""
    patient_ids = list(_oracle.all_patient_ids)
    persona_map = _oracle.persona_map
    # Try to load params from patient_params directory
    try:
        theoretical_params = _oracle.load_patient_params(patient_ids, _oracle.apps, _oracle.PARAM_DIR)
    except FileNotFoundError:
        typer.echo("Parameter files not found; generating default parameters first.")
        create_params(similarity_alpha=1.0, out_dir=_oracle.PARAM_DIR)
        theoretical_params = _oracle.load_patient_params(patient_ids, _oracle.apps, _oracle.PARAM_DIR)

    df = _oracle.generate_full_event_stream_sessions(num_weeks, patient_ids, persona_map, theoretical_params)
    df.to_csv(out_csv, index=False)
    typer.echo(f"Saved events to {out_csv}")
