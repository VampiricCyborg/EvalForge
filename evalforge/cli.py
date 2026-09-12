"""CLI entrypoint for EvalForge.

Not yet implemented — this will grow subcommands for running evals,
generating reports, and comparing runs.
"""

import argparse

from evalforge import __version__


def main() -> None:
    parser = argparse.ArgumentParser(prog="evalforge")
    parser.add_argument("--version", action="version", version=__version__)
    parser.parse_args()


if __name__ == "__main__":
    main()
