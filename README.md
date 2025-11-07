# my_research_tool

Small CLI wrapper around the existing simulation and analysis code.

Install (editable):

```sh
pip install -e .
```

Usage (after installing):

```sh
my_tool --help
```

This project places the CLI in `src/my_tool/cli.py` and provides commands to:
- create theoretical parameters
- generate simulation event data
- run lightweight analysis
- produce validation visualizations

Refer to `pyproject.toml` for dependencies.
