"""Dry run of the regression-metrics notebook (kaggle/regression/regression_source.py) on
small synthetic data (CPU), with fake checkpoints for all 17 trained models laid out like the
real Kaggle inputs. Checks code paths and the metric arithmetic, not real numbers.
No competition data is used.

Usage: python build_regression.py && python dry_run_regression.py
"""
import os, sys, json, tempfile, importlib.util
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
T, NT, H, W = 160, 4, 140, 124
root = tempfile.mkdtemp()
os.chdir(root)
spec = importlib.util.spec_from_file_location('regression', os.path.join(HERE, '..', 'notebooks', 'kaggle', 'regression', 'regression_source.py'))
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
    np.save(os.path.join(data, 'test_in', f'{f}.npy'), np.stack([field(f, C.T_IN, h0=5 * i) for i in range(NT)]))
ll = np.stack(np.meshgrid(np.linspace(8, 37, H), np.linspace(68, 97, W), indexing='ij'), -1)
np.save(os.path.join(data, 'raw', 'lat_long.npy'), ll.astype(np.float32))

inp = os.path.join(root, 'inputs')
names = ('v4', 'r1', 'fmg', 'cl', 'pa', 'final', 'seeds_resgru', 'seeds_afnov4', 'abl_a', 'abl_b', 'analysis')
dirs = {k: os.path.join(inp, k) for k in names}
for d in dirs.values():
    os.makedirs(d)
for m in C.MONTHS:
    np.savez_compressed(os.path.join(dirs['v4'], f'ep_mask_{m}.npz'),
                        mask=(rng.random((T, H, W)) < 0.03).astype(np.uint8), sigma=np.ones((H, W), np.float32))
C.DATA_ROOT, C.RAW, C.TEST = data, os.path.join(data, 'raw'), os.path.join(data, 'test_in')
C.V4_DIR, C.N_TEST, C.WORKERS = dirs['v4'], NT, 0
C.WORK = os.path.join(root, 'build')
os.makedirs(C.WORK)
stats = S.build_stats()
os.replace(os.path.join(C.WORK, 'stats.npz'), os.path.join(dirs['v4'], 'stats.npz'))
months, static = S.load_months(stats)
val = S.PM25Windows(months, static, stats, 'val')
test = S.PM25Test(stats, static)


def fake(bb, seed, d, name, summary=None):
    torch.manual_seed(seed)
    model = S.BACKBONES[bb]()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.01 * torch.randn_like(p))
    score = S.evaluate(model, torch.utils.data.DataLoader(val, batch_size=4))['score']
    torch.save({'model': model.state_dict(), 'metrics': {'score': score}, 'epoch': 1}, os.path.join(d, name))
    if summary:
        json.dump(dict(seed=seed, secs_per_epoch=100.0 + seed), open(os.path.join(d, summary), 'w'))
        np.save(os.path.join(d, f'preds_s{seed}.npy'), S.predict(model, test, 4))
    return model


fake('afnov4', 42, dirs['v4'], 's42_phaseA_best.pt')
fake('afnov4', 1337, dirs['v4'], 's1337_phaseA_best.pt')
fake('resgru', 42, dirs['r1'], 'resgru_s42_phaseA_best.pt')
json.dump(dict(seed=42, secs_per_epoch=189.0), open(os.path.join(dirs['r1'], 'summary.json'), 'w'))
fake('resgru', 1337, dirs['final'], 'resgru_s1337_phaseA_best.pt')
for s in (1, 2, 3):
    fake('resgru', s, dirs['seeds_resgru'], f'resgru_s{s}_phaseA_best.pt', f'summary_s{s}.json')
    fake('afnov4', s, dirs['seeds_afnov4'], f'afnov4_s{s}_phaseA_best.pt', f'summary_s{s}.json')
for bb, d in (('fmgwaclu', 'fmg'), ('convlstm', 'cl'), ('patchafno', 'pa')):
    fake(bb, 42, dirs[d], f'{bb}_s42_phaseA_best.pt')
for bb, d in (('afnov4_allskip', 'abl_a'), ('afnov4_nosparse', 'abl_a'), ('afnov4_pad', 'abl_b'), ('resgru_norec', 'abl_b')):
    fake(bb, 42, dirs[d], f'{bb}_s42_phaseA_best.pt')

# reference files: analysis.json from the real analysis code, final-round values computed independently here
S.INPUT_ROOT, S.N_BOOT = inp, 20
C.WORK = dirs['analysis']
S.analyze()
fr = {}
for f in S.FAMS:
    preds = []
    for s in (42, 1337):
        mdl = S.BACKBONES[f]()
        mdl.load_state_dict(torch.load(S.ckpt_of(f, s), map_location='cpu', weights_only=False)['model'])
        p, tg, mk = S.val_arrays(mdl, val)
        preds.append(p)
    e = (preds[0] + preds[1]) / 2
    sc = S.fit_scales(e, tg, mk)
    fr[f] = dict(two_seed=S.terms_of(S.wh_terms(e, tg, mk)),
                 two_seed_cal_in_sample=S.terms_of(S.wh_terms(e * sc[None, :, None, None], tg, mk)))
json.dump(dict(seed=1337, secs_per_epoch=186.2, final_round=fr), open(os.path.join(dirs['final'], 'summary.json'), 'w'))
del months

S.INPUT_ROOT = os.path.join(root, 'empty')
os.makedirs(S.INPUT_ROOT)
try:
    S.regression()
    stopped = False
except RuntimeError:
    stopped = True
check(stopped, 'pre-check stops when inputs are missing')

S.INPUT_ROOT, S.PERSIST_REF = inp, None
C.WORK = os.path.join(root, 'work_regression')
os.makedirs(C.WORK)
S.regression()
r = json.load(open(os.path.join(C.WORK, 'regression_metrics.json')))
M = r['models']
check(r['all_checks_pass'], f'all built-in checks pass: {r["checks"]}')
check(len(r['checks']) == 17 + 10 + 1, f'28 built-in checks (17 logged, 10 reference, 1 self-fit), got {len(r["checks"])}')
check(len(M) == 30, f'30 models (17 trained, 12 means, 1 persistence), got {len(M)}')
need = ('mae', 'rmse', 'r2', 'mae_log1p', 'rmse_log1p', 'r2_log1p', 'r_pooled', 'score', 'gsmape', 'esmape', 'ecorr')
check(all(all(k in m for k in need) and all(len(m['per_hour'][x]) == 16 for x in ('mae', 'rmse', 'r2'))
          and m['episode']['n'] + m['other']['n'] == r['truth']['n'] for m in M.values()),
      'every model has the metrics, 16 per-hour values each, and episode + other = all values')

mdl = S.BACKBONES['convlstm']()
mdl.load_state_dict(torch.load(S.ckpt_of('convlstm', 42), map_location='cpu', weights_only=False)['model'])
p, y, mk = S.val_arrays(mdl, val)
p, y = p.astype(np.float64), y.astype(np.float64)
mae, rmse = np.abs(p - y).mean(), np.sqrt(((p - y) ** 2).mean())
r2 = 1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()
r2h0 = 1 - ((p[:, 0] - y[:, 0]) ** 2).sum() / ((y[:, 0] - y[:, 0].mean()) ** 2).sum()
r2l = 1 - ((np.log1p(p) - np.log1p(y)) ** 2).sum() / ((np.log1p(y) - np.log1p(y).mean()) ** 2).sum()
rp = np.corrcoef(p.ravel(), y.ravel())[0, 1]
c = M['convlstm_s42']
check(abs(c['mae'] - mae) < 1e-3 and abs(c['rmse'] - rmse) < 1e-3, f'MAE/RMSE match an independent recomputation ({c["mae"]} vs {mae:.4f}, {c["rmse"]} vs {rmse:.4f})')
check(abs(c['r2'] - r2) < 1e-4 and abs(c['per_hour']['r2'][0] - r2h0) < 1e-4 and abs(c['r2_log1p'] - r2l) < 1e-4,
      f'R2 (pooled, hour 1, log1p) match ({c["r2"]} vs {r2:.5f}; {c["per_hour"]["r2"][0]} vs {r2h0:.5f}; {c["r2_log1p"]} vs {r2l:.5f})')
check(abs(c['r_pooled'] - rp) < 1e-4, f'pooled Pearson r matches np.corrcoef ({c["r_pooled"]} vs {rp:.5f})')

pers = S.val_persistence(val)
mi, s0 = val.samples[0]
raw = np.load(os.path.join(C.RAW, val.keys[mi], 'cpm25.npy'))[s0 + C.T_IN - 1]
check(np.allclose(pers[0, 0], raw, rtol=1e-4, atol=1e-4) and np.allclose(pers[0, 15], raw, rtol=1e-4, atol=1e-4),
      'persistence = the raw last input hour, repeated for all 16 hours')
check(M['persistence']['stage'] == 'baseline' and M['resgru_5seed_cal_lomo']['stage'] == '5 seeds'
      and M['afnov4_2seed_cal_in']['stage'] == 'final round (2 seeds)' and M['resgru_norec_s42']['stage'] == 'ablation (1 seed)',
      'stages labelled')
check(set(r['single_seed_summary']) == set(S.FAMS) and 'sd' in r['single_seed_summary']['resgru']['mae'], 'single-seed mean and sd per family')
print('=' * 70)
print('DRY RUN PASSED' if not FAIL else f'{len(FAIL)} CHECK(S) FAILED: {FAIL}')
sys.exit(1 if FAIL else 0)
