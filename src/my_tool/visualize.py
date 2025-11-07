"""Visualization helpers that wrap validation plotting from `full_pipeline.py`."""
import os
import typer

try:
    from ... import full_pipeline as _full
except Exception:
    import full_pipeline as _full


def run_validation(patient_id: int = None):
    """Run validation dashboards. If patient_id is None, run for hero patients as in full_pipeline main."""
    # Load params and events as the full_pipeline main does
    if not os.path.exists(_full.EVENT_CACHE_PATH):
        typer.echo(f"Event cache '{_full.EVENT_CACHE_PATH}' not found. Generate data first.")
        return

    # Reuse main logic for hero selection when patient_id is None
    try:
        theoretical_params = _full.load_patient_params(_full.all_patient_ids, _full.apps, _full.PARAM_DIR)
    except Exception as e:
        typer.echo(f"Error loading params: {e}")
        return

    event_df = _full.pd.read_csv(_full.EVENT_CACHE_PATH)
    event_df['timestamp'] = _full.pd.to_datetime(event_df['timestamp'])
    transitions_df = _full._get_transitions(event_df)

    if patient_id is not None:
        _full.plot_single_patient_dashboard(patient_id, transitions_df, theoretical_params, _full.persona_map, _full.schedules, _full.apps)
    else:
        # mimic hero selection in full_pipeline
        import random
        personas_to_patients = {}
        for patient, persona in _full.persona_map.items():
            personas_to_patients.setdefault(persona, []).append(patient)
        hero_patients = [random.choice(p) for p in personas_to_patients.values()]
        for pid in hero_patients:
            _full.plot_single_patient_dashboard(pid, transitions_df, theoretical_params, _full.persona_map, _full.schedules, _full.apps)
