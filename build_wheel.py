#!/usr/bin/env python3
"""Utility script to build the project wheel.

This script invokes ``python -m build`` with the ``--wheel`` option. The
``build`` package must be installed in the environment.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys


def main() -> None:
    project_root = pathlib.Path(__file__).resolve().parent
    cmd = [sys.executable, "-m", "build", "--wheel", str(project_root)]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
