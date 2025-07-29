#!/usr/bin/env python3
import subprocess
from pathlib import Path
import tomllib


def build_wheels():
    project_dir = Path(__file__).resolve().parents[1]
    pyproject_file = project_dir / 'pyproject.toml'
    with pyproject_file.open('rb') as f:
        project_data = tomllib.load(f)

    deps = project_data.get('project', {}).get('dependencies', [])

    wheel_dir = project_dir / 'wheelhouse'
    wheel_dir.mkdir(exist_ok=True)

    # Build wheel for the project itself
    subprocess.check_call([
        'python', '-m', 'pip', 'wheel', str(project_dir),
        '--wheel-dir', str(wheel_dir), '--no-deps'
    ])

    # Build wheels for each dependency
    for dep in deps:
        subprocess.check_call([
            'python', '-m', 'pip', 'wheel', dep,
            '--wheel-dir', str(wheel_dir)
        ])


if __name__ == '__main__':
    build_wheels()
