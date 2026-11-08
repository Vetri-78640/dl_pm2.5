"""Dry run of the ridge notebook on small synthetic data (CPU). No competition data.
Usage: python build_ridge.py && python dry_run_ridge.py
"""
import os, sys, json, tempfile, importlib.util
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
T, H, W = 160, 140, 124
root = tempfile.mkdtemp()
os.chdir(root)
spec = importlib.util.spec_from_file_location('bl', os.path.join(HERE, '..', 'notebooks', 'kaggle', 'ridge', 'ridge_source.py'))
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
os.makedirs(os.path.join(data, 'test_in'))
for f in C.FEATS:
    np.save(os.path.join(data, 'test_in', f'{f}.npy'), np.stack([field(f)[:C.T_IN] for _ in range(3)]))
C.DATA_ROOT, C.RAW, C.TEST, C.V4_DIR, C.WORKERS, C.N_TEST = data, os.path.join(data, 'raw'), os.path.join(data, 'test_in'), v4, 0, 3
C.WORK = os.path.join(root, 'work_bl')
os.makedirs(C.WORK)
S.ridge()
r = json.load(open(os.path.join(C.WORK, 'ridge.json')))
check(r['all_checks_pass'], f'built-in checks pass: {r["checks"]}')
p = np.load(os.path.join(C.WORK, 'preds_ridge.npy'))
check(p.shape == (3, H, W, 16) and np.isfinite(p).all(), f'test preds {p.shape}')
check(len(r['weights']) == 30 and len(r['weights'][0]) == 16 and set(r['lambda_grid']) == {'0.01', '1.0', '100.0', '10000.0', '1000000.0', 'zero'},
      '30 x 16 weights and the full lambda grid')
x = np.random.default_rng(1).normal(size=(20, C.T_IN, H, W))
f = S.ridge_features(x)
check(f.shape == (H * W, 30) and np.allclose(f[:, :20], x[:, -1].reshape(20, -1).T) and np.allclose(f[:, 20:29], x[0, :-1].reshape(9, -1).T) and (f[:, 29] == 1).all(),
      'features = 20 channels at the last hour + PM channel at the 9 earlier hours + bias')
m = r['model']
check(all(k in m for k in ('mae', 'rmse', 'r2', 'score', 'per_hour')), 'ridge scored with the regression metrics')
print('DRY RUN PASSED' if not FAIL else f'{len(FAIL)} FAILED: {FAIL}')
sys.exit(1 if FAIL else 0)
