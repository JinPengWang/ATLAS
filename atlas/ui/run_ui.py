"""CLI entry point for launching the ATLAS Web Console."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Launch the ATLAS Streamlit dashboard from the command line."""
    dashboard_path = Path(__file__).parent / "dashboard.py"

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(dashboard_path),
    ] + sys.argv[1:]

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nStopping ATLAS Web Console...")


if __name__ == "__main__":
    main()
