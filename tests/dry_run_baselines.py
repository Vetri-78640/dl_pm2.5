"""Dry run of the baselines notebook on small synthetic data (CPU). No competition data.
Usage: python build_baselines.py && python dry_run_baselines.py
"""
import os, sys, json, tempfile, importlib.util
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
T, H, W = 160, 140, 124
root = tempfile.mkdtemp()
os.chdir(root)
spec = importlib.util.spec_from_file_location('bl', os.path.join(HERE, '..', 'notebooks', 'kaggle', 'baselines', 'baselines_source.py'))
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
C = S.C
rng = np.random.default_rng(0)
FAIL = []


def check(ok, msg):
    print(('PASS ' if ok else 'FAIL ') + msg)
    if not ok:
        FAIL.append(msg)


def field(kind):
    d = np.clip(np.sin(2 * np.pi * (np.arange(T) % 24 - 6) / 24), 0, None)[:, None, None]
    return {'cpm25': rng.lognormal(3.5, 0.5, (T, H, W)), 'q2': 0.01 + 0.002 * rng.random((T, H, W)),
            't2': 295 + 8 * d + rng.normal(0, 1, (T, H, W)), 'u10': rng.normal(0, 3, (T, H, W)),
            'v10': rng.normal(0, 3, (T, H, W)), 'pblh': 200 + 1000 * d + 50 * rng.random((T, H, W)),
            'psfc': 1e5 + 100 * rng.normal(0, 1, (T, H, W)), 'swdown': 800 * d * (0.9 + 0.1 * rng.random((T, H, W))),
            'rain': rng.exponential(0.1, (T, H, W))}[kind].astype(np.float32)


data, v4, ref = (os.path.join(root, k) for k in ('data', 'inputs/v4', 'inputs/reg'))
for m in C.MONTHS:
    os.makedirs(os.path.join(data, 'raw', m))
    for f in C.FEATS:
        np.save(os.path.join(data, 'raw', m, f'{f}.npy'), field(f))
ll = np.stack(np.meshgrid(np.linspace(8, 37, H), np.linspace(68, 97, W), indexing='ij'), -1)
np.save(os.path.join(data, 'raw', 'lat_long.npy'), ll.astype(np.float32))
os.makedirs(v4)
os.makedirs(ref)
for m in C.MONTHS:
    np.savez_compressed(os.path.join(v4, f'ep_mask_{m}.npz'), mask=(rng.random((T, H, W)) < 0.03).astype(np.uint8),
                        sigma=np.ones((H, W), np.float32))
C.DATA_ROOT, C.RAW, C.V4_DIR, C.WORKERS = data, os.path.join(data, 'raw'), v4, 0
C.WORK = os.path.join(root, 'work_bl')
os.makedirs(C.WORK)
S.INPUT_ROOT = os.path.join(root, 'inputs')
try:
    S.baselines()
    stopped = False
except RuntimeError:
    stopped = True
check(stopped, 'stops when the regression output is missing')

stats = S.build_stats()
months, static = S.load_months(stats)
val = S.PM25Windows(months, static, stats, 'val')
tgt, msk, per = S.val_truth(val)
m = S.reg_metrics(per, tgt, msk, S.truth_stats(tgt, msk))
m.update(S.terms_of(S.wh_terms(per, tgt, msk)))
json.dump({'models': {'persistence': m}}, open(os.path.join(ref, 'regression_metrics.json'), 'w'))

S.baselines()
r = json.load(open(os.path.join(C.WORK, 'baselines.json')))
check(r['all_checks_pass'], f'built-in checks pass: {r["checks"]}')
check(set(r['models']) == {'persistence', 'seasonal_naive'}, 'two baselines scored')
mi, s0 = val.samples[0]
rawm = np.load(os.path.join(C.RAW, val.keys[mi], 'cpm25.npy'))
sn, _ = S.seasonal_naive(val, {k: np.load(os.path.join(C.RAW, k, 'cpm25.npy')) for k in val.keys})
check(np.array_equal(sn[0, 0], rawm[s0 + C.T_IN - 24]) and np.array_equal(sn[0, 15], rawm[s0 + C.T_IN - 24 + 15]),
      'seasonal naive = the same pixel 24 h before each target hour (leads 1 and 16)')
check(np.allclose(tgt[0, 0], rawm[s0 + C.T_IN], rtol=1e-4, atol=1e-4), 'target hour 1 = raw hour s0 + 10')
print('DRY RUN PASSED' if not FAIL else f'{len(FAIL)} FAILED: {FAIL}')
sys.exit(1 if FAIL else 0)
