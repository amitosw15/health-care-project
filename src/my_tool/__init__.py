"""my_tool package entrypoint.

Importing the package should be lightweight. Importing the full CLI (which
can pull heavy scientific dependencies) is done lazily via :func:`get_app`.
"""

from typing import Any


def get_app() -> Any:
	"""Lazily import and return the Typer app.

	Use this in entrypoints (``if __name__ == '__main__'``) or when you want
	to run the CLI programmatically. Avoids importing heavy modules on plain
	`import my_tool`.
	"""
	from .cli import app
	return app


__all__ = ["get_app"]
