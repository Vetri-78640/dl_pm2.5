"""Dry run of the 5-seed analysis notebook (kaggle/analysis/analysis_source.py) on
small synthetic data (CPU) with 10 fake checkpoints laid out like the real Kaggle
inputs. Checks code paths, not numbers. No competition data is used.

Usage: python build_analysis.py && python dry_run_analysis.py
"""
import os, sys, json, tempfile, importlib.util
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
T, NT, H, W = 160, 4, 140, 124
root = tempfile.mkdtemp()
os.chdir(root)
spec = importlib.util.spec_from_file_location('analysis', os.path.join(HERE, '..', 'notebooks', 'kaggle', 'analysis', 'analysis_source.py'))
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
C = S.C
rng = np.random.default_rng(0)
FAIL = []


def check(ok, msg):
    print(('PASS ' if ok else 'FAIL ') + msg)
    if not ok:
        FAIL.append(msg)


def field(kind, n_t, h0=0):
    d = np.clip(np.sin(2 * np.pi * ((h0 + np.arange(n_t)) % 24 - 6) / 24), 0, None)
    base = {'cpm25': lambda: rng.lognormal(3.5, 0.5, (n_t, H, W)),
            'q2': lambda: 0.01 + 0.002 * rng.random((n_t, H, W)),
            't2': lambda: 295 + 8 * d[:, None, None] + rng.normal(0, 1, (n_t, H, W)),
            'u10': lambda: rng.normal(0, 3, (n_t, H, W)),
            'v10': lambda: rng.normal(0, 3, (n_t, H, W)),
            'pblh': lambda: 200 + 1000 * d[:, None, None] + 50 * rng.random((n_t, H, W)),
            'psfc': lambda: 1e5 + 100 * rng.normal(0, 1, (n_t, H, W)),
            'swdown': lambda: 800 * d[:, None, None] * (0.9 + 0.1 * rng.random((n_t, H, W))),
            'rain': lambda: rng.exponential(0.1, (n_t, H, W))}[kind]
    return base().astype(np.float32)


data = os.path.join(root, 'data')
for m in C.MONTHS:
    os.makedirs(os.path.join(data, 'raw', m))
    for f in C.FEATS:
        np.save(os.path.join(data, 'raw', m, f'{f}.npy'), field(f, T))
os.makedirs(os.path.join(data, 'test_in'))
for f in C.FEATS:
    np.save(os.path.join(data, 'test_in', f'{f}.npy'),
            np.stack([field(f, C.T_IN, h0=5 * i) for i in range(NT)]))
ll = np.stack(np.meshgrid(np.linspace(8, 37, H), np.linspace(68, 97, W), indexing='ij'), -1)
np.save(os.path.join(data, 'raw', 'lat_long.npy'), ll.astype(np.float32))

inp = os.path.join(root, 'inputs')
dirs = {k: os.path.join(inp, k) for k in ('v4', 'r1', 'final', 'seeds_resgru', 'seeds_afnov4')}
for d in dirs.values():
    os.makedirs(d)
for m in C.MONTHS:
    np.savez_compressed(os.path.join(dirs['v4'], f'ep_mask_{m}.npz'),
                        mask=(rng.random((T, H, W)) < 0.03).astype(np.uint8),
                        sigma=np.ones((H, W), np.float32))
C.DATA_ROOT, C.RAW, C.TEST = data, os.path.join(data, 'raw'), os.path.join(data, 'test_in')
C.V4_DIR, C.N_TEST, C.WORKERS = dirs['v4'], NT, 0
C.WORK = os.path.join(root, 'build')
os.makedirs(C.WORK)
stats = S.build_stats()
os.replace(os.path.join(C.WORK, 'stats.npz'), os.path.join(dirs['v4'], 'stats.npz'))
months, static = S.load_months(stats)
val = S.PM25Windows(months, static, stats, 'val')
test = S.PM25Test(stats, static)


def fake(fam, seed, d, name, summary=None):
    torch.manual_seed(seed)
    model = S.make(fam)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.01 * torch.randn_like(p))
    score = S.evaluate(model, torch.utils.data.DataLoader(val, batch_size=4))['score']
    torch.save({'model': model.state_dict(), 'metrics': {'score': score}, 'epoch': 1}, os.path.join(d, name))
    if summary:
        json.dump(dict(seed=seed, secs_per_epoch=100.0 + seed), open(os.path.join(d, summary), 'w'))
        np.save(os.path.join(d, f'preds_s{seed}.npy'), S.predict(model, test, 4))


fake('afnov4', 42, dirs['v4'], 's42_phaseA_best.pt')
fake('afnov4', 1337, dirs['v4'], 's1337_phaseA_best.pt')
fake('resgru', 42, dirs['r1'], 'resgru_s42_phaseA_best.pt')
json.dump(dict(seed=42, secs_per_epoch=189.0), open(os.path.join(dirs['r1'], 'summary.json'), 'w'))
fake('resgru', 1337, dirs['final'], 'resgru_s1337_phaseA_best.pt')
json.dump(dict(seed=1337, secs_per_epoch=186.2), open(os.path.join(dirs['final'], 'summary.json'), 'w'))
for s in (1, 2, 3):
    fake('resgru', s, dirs['seeds_resgru'], f'resgru_s{s}_phaseA_best.pt', f'summary_s{s}.json')
    fake('afnov4', s, dirs['seeds_afnov4'], f'afnov4_s{s}_phaseA_best.pt', f'summary_s{s}.json')
del months

S.INPUT_ROOT = os.path.join(root, 'empty')
os.makedirs(S.INPUT_ROOT)
try:
    S.precheck()
    stopped = False
except RuntimeError:
    stopped = True
check(stopped, 'pre-check stops when checkpoints are missing')

S.INPUT_ROOT, S.N_BOOT = inp, 50
C.WORK = os.path.join(root, 'work_analysis')
os.makedirs(C.WORK)
S.analyze()
r = json.load(open(os.path.join(C.WORK, 'analysis.json')))
check(r['all_checks_pass'], f'all built-in checks pass: {r["checks"]}')
check(len(r['checks']) == 16, f'16 checks (10 val, 6 test-pred), got {len(r["checks"])}')
lo, hi = r['gap']['ensemble_ci95']
check(lo <= hi, f'bootstrap interval ordered {lo} <= {hi}')
check(set(r['per_hour']['resgru']['seeds']) == {'42', '1337', '1', '2', '3'}, 'per-hour terms for all 5 seeds')
check(all(len(r['per_hour'][f]['ensemble']['score']) == 16 for f in S.FAMS), 'per-hour ensemble scores for 16 hours')
check(len(r['bins']['resgru']) == 6 and len(r['cell_deciles']) == 10, 'six concentration bins, ten cell deciles')
check(set(r['calibration']['afnov4']['scales_lomo']) == set(C.MONTHS), 'leave-one-month-out scales for all 4 months')
check(r['seeds_val']['resgru']['1337']['secs_per_epoch'] == 186.2, 'secs/epoch read from the final-round summary')
check(r['cost']['afnov4']['params'] > 0 and r['cost']['resgru']['ms_per_window'] > 0, 'cost measured')
for f in ('fig_horizon.png', 'fig_bins.png', 'fig_maps.png', 'fig_case.png'):
    check(os.path.getsize(os.path.join(C.WORK, f)) > 10000, f'{f} written')
for f in ('preds_afnov4_5seed.npy', 'preds_afnov4_5seed_cal.npy', 'preds_resgru_5seed.npy',
          'preds_resgru_5seed_cal.npy', 'preds_resgru_s1337.npy'):
    p = np.load(os.path.join(C.WORK, f))
    check(p.shape == (NT, H, W, 16) and np.isfinite(p).all(), f'{f} {p.shape}, finite')
e = np.load(os.path.join(C.WORK, 'preds_resgru_5seed.npy'))
ec = np.load(os.path.join(C.WORK, 'preds_resgru_5seed_cal.npy'))
sc = np.array(r['calibration']['resgru']['scales_in_sample'], np.float32)
check(np.allclose(ec, e * sc[None, None, None, :], rtol=1e-3), 'calibrated preds = 5-seed preds x in-sample scales')
print('=' * 70)
print('DRY RUN PASSED' if not FAIL else f'{len(FAIL)} CHECK(S) FAILED: {FAIL}')
print('work folder:', C.WORK)
sys.exit(1 if FAIL else 0)
