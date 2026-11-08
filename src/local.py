"""Shared setup for the local notebooks in notebooks/.

load_study() imports the code exactly as it ran on Kaggle (notebooks/kaggle/ridge/ridge_source.py
holds the pipeline, all networks, the analysis, the metrics, the baselines and ridge) and points it at
the local data, the V4 episode masks and statistics, and a scratch folder.
"""
import importlib.util
import json
import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(ROOT, 'notebooks', 'kaggle', 'ridge', 'ridge_source.py')


def config():
    return yaml.safe_load(open(os.path.join(ROOT, 'configs', 'study.yaml')))


def path(*parts):
    return os.path.join(ROOT, *parts)


def result(*parts):
    return path(config()['paths']['results'], *parts)


def load_json(*parts):
    return json.load(open(result(*parts)))


def restore_backend(backend):
    """The as-run module switches matplotlib to Agg; switch back so notebooks show figures inline."""
    import matplotlib
    try:
        from IPython import get_ipython
        ip = get_ipython()
    except ImportError:
        ip = None
    if ip is not None and 'inline' in backend:
        ip.run_line_magic('matplotlib', 'inline')
    else:
        matplotlib.use(backend)


def load_study(with_data=True):
    import matplotlib
    backend = matplotlib.get_backend()
    work = path('data', 'processed')
    os.makedirs(work, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(work)  # the module creates ./work when imported
    try:
        spec = importlib.util.spec_from_file_location('study_as_run', SOURCE)
        S = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(S)
    finally:
        os.chdir(cwd)
        restore_backend(backend)
    C = S.C
    C.WORK = os.path.join(work, 'work')
    C.WORKERS = 0
    S.INPUT_ROOT = None
    v4 = path('data', 'interim', 'v4')
    C.V4_DIR = v4 if os.path.isdir(v4) else None
    if with_data:
        root = S.find_dir(path(config()['paths']['data']), ['raw/APRIL_16/cpm25.npy', 'test_in/cpm25.npy'])
        if root is None:
            raise FileNotFoundError('competition data not found under data/raw; see data/README.md')
        C.DATA_ROOT, C.RAW, C.TEST = root, os.path.join(root, 'raw'), os.path.join(root, 'test_in')
    return S
