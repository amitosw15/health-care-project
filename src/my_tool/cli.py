import typer
from typing import Optional

# A very small, hard-coded set of personas for the lightweight `info` command.
# Keeping this here avoids importing `oracle.py` just to show available personas.
PERSONAS = [
    '9-to-5er',
    'Night Owl',
    'Influencer',
    'Compulsive Checker',
]


app = typer.Typer(help="My research tool CLI (wraps existing scripts)")


@app.command()
def info():
    """Print lightweight info about the tool (no heavy imports)."""
    typer.echo("my_research_tool CLI")
    typer.echo(f"Available (example) personas: {', '.join(PERSONAS)}")
    typer.echo("Commands: create-params, generate-data, analyze, validate")


@app.command()
def create_params(alpha: float = 1.0, out_dir: str = "patient_params"):
    """Generate and save patient theoretical parameters.

    This imports the heavier `data` wrapper lazily when invoked.
    """
    typer.echo(f"Generating params with alpha={alpha} -> dir={out_dir}")
    # Lazy import to avoid bringing in heavy dependencies at module-import time
    from . import data as _data
    _data.create_params(similarity_alpha=alpha, out_dir=out_dir)


@app.command()
def generate_data(num_weeks: int = 4, out_csv: str = "event_data_cache_sessions_distinct.csv"):
    """Generate event data and save to CSV."""
    typer.echo(f"Generating event data for {num_weeks} weeks -> {out_csv}")
    from . import data as _data
    _data.generate_event_data(num_weeks=num_weeks, out_csv=out_csv)


@app.command()
def analyze(experiment: Optional[str] = None):
    """Run a lightweight analysis/experiment. If `experiment` is None a default run is used."""
    typer.echo(f"Running analysis (experiment={experiment})")
    from . import analysis as _analysis
    _analysis.run_analysis(experiment_name=experiment)


@app.command()
def validate(patient: Optional[int] = None):
    """Generate validation dashboard(s). If patient is None, generate for sample hero patients."""
    typer.echo(f"Generating validation dashboard (patient={patient})")
    from . import visualize as _viz
    _viz.run_validation(patient_id=patient)


def main():
    app()


if __name__ == "__main__":
    main()
