"""Runs each synthetic-data dry run and the port check (about 35 min in total on a laptop CPU).

Slow, so they only run on request:  pytest -m dryrun
The port check reads the original notebooks in reference/originals/.
"""
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRY_RUNS = sorted(f for f in os.listdir(os.path.join(ROOT, 'tests')) if f.startswith('dry_run') and f.endswith('.py'))


@pytest.mark.dryrun
@pytest.mark.parametrize('script', DRY_RUNS)
def test_dry_run(script):
    r = subprocess.run([sys.executable, os.path.join(ROOT, 'tests', script)], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]


@pytest.mark.dryrun
def test_port_check():
    r = subprocess.run([sys.executable, os.path.join(ROOT, 'tests', 'local_check.py'), os.path.join(ROOT, 'reference', 'originals')],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
