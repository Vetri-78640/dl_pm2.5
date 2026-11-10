"""Dry run of the hybrid notebook on small synthetic data (CPU). No competition data.
Usage: python build_hybrid.py && python dry_run_hybrid.py
"""
import os, sys, json, tempfile, importlib.util
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
T, H, W = 160, 140, 124
root = tempfile.mkdtemp()
os.chdir(root)
spec = importlib.util.spec_from_file_location('bl', os.path.join(HERE, '..', 'notebooks', 'kaggle', 'hybrid', 'hybrid_source.py'))
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
C.WORK = os.path.join(root, 'work_hy')
os.makedirs(C.WORK)
for s in S.SEEDS5:
    torch.manual_seed(s)
    model = S.BACKBONES['resgru']()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.01 * torch.randn_like(p))
    torch.save({'model': model.state_dict()}, os.path.join(ref, f'resgru_s{s}_phaseA_best.pt'))
S.INPUT_ROOT = os.path.join(root, 'inputs')
S.hybrid()
r = json.load(open(os.path.join(C.WORK, 'hybrid.json')))
check(r['all_checks_pass'], f'built-in checks pass: {r["checks"]}')
check(set(r['folds']) == set(C.MONTHS) and all(f['n_test_windows'] > 0 and f['n_fit_windows'] + f['n_test_windows'] == r['n_val_windows']
                                                for f in r['folds'].values()), 'four folds, each holds out one whole month')
wins = sum(f['hybrid']['score'] > f['ensemble']['score'] for f in r['folds'].values())
exp = 'better' if r['pooled']['hybrid']['score'] > r['pooled']['ensemble']['score'] and wins >= 3 else 'no gain'
check(r['verdict'] == exp and r['months_won'] == wins, f'verdict follows the rule: {r["verdict"]} ({wins}/4 months)')
check(os.path.exists(os.path.join(C.WORK, 'preds_hybrid.npy')) == (r['verdict'] == 'better'), 'test predictions written only if the verdict is better')
check(len(r['features']) == 33 and len(r['importance_gain_share_all_months_fit']) == 33, '33 named features')

# independent checks of the pieces
g = np.random.default_rng(1)
x = g.normal(size=(20, C.T_IN, H, W)).astype(np.float32)
ll = g.normal(3, 0.5, (H, W)).astype(np.float32)
bl = g.normal(3, 0.5, (16, H, W)).astype(np.float32)
sp = g.random((16, H, W)).astype(np.float32)
rows = S.hybrid_rows(x, ll, bl, sp)
j, h = 5 * W + 7, 11
row = rows[h * H * W + j]
check(rows.shape == (16 * H * W, 33) and np.allclose(row[:20], x[:, -1, 5, 7]) and np.allclose(row[20:29], x[0, :-1, 5, 7])
      and np.isclose(row[29], bl[h, 5, 7]) and np.isclose(row[30], bl[h, 5, 7] - ll[5, 7]) and np.isclose(row[31], sp[h, 5, 7]) and row[32] == h + 1,
      'row layout: 29 input features, base, base change, seed spread, lead hour')
sel = S.hybrid_sel()
check(sel.sum() == 35 * 31 and np.array_equal(S.hybrid_rows(x, ll, bl, sp, sel), rows.reshape(16, H * W, 33)[:, sel].reshape(-1, 33)),
      'pixel subsample is every 4th pixel in each direction')
check(np.allclose(S.hybrid_apply(None, None, None, bl, None), np.expm1(bl), rtol=1e-5), 'zero correction returns the base forecast')


class Const:
    def predict(self, X):
        return np.full(len(X), np.log(2.0))


check(np.allclose(S.hybrid_apply(Const(), x, ll, bl, sp), 2 * (np.expm1(bl) + 1) - 1, rtol=1e-4), 'a log correction of log 2 doubles 1 + forecast')

# test path, whatever the verdict was
stats = S.build_stats()
months, static = S.load_months(stats)
val = S.PM25Windows(months, static, stats, 'val')
base, base_log, spread, tgt, msk = S.resgru_ensemble(val)
ps = []
for s in S.SEEDS5:
    m = S.BACKBONES['resgru']()
    m.load_state_dict(torch.load(S.ckpt_of('resgru', s), map_location='cpu', weights_only=False)['model'])
    ps.append(S.val_arrays(m, val)[0])
check(np.allclose(base, np.mean(ps, 0), rtol=1e-5, atol=1e-5) and np.allclose(spread, np.std(np.log1p(np.array(ps, np.float64)), 0), atol=1e-4),
      'ensemble = mean of the 5 seeds; spread = sd of their log forecasts')
model, n = S.hybrid_fit(val, np.arange(len(val)), base_log, spread, tgt)
check(n == len(val) * 16 * 35 * 31, f'fit rows = windows x 16 x 1085 ({n:,})')
test = S.PM25Test(stats, static)
_, tl, tsp, _, _ = S.resgru_ensemble(test, test=True)
tp = np.stack([S.hybrid_apply(model, *test[i], tl[i], tsp[i]) for i in range(len(test))]).transpose(0, 2, 3, 1)
check(tp.shape == (3, H, W, 16) and np.isfinite(tp).all() and tp.min() >= 0, f'test path gives {tp.shape}, finite, non-negative')
print('DRY RUN PASSED' if not FAIL else f'{len(FAIL)} FAILED: {FAIL}')
sys.exit(1 if FAIL else 0)
