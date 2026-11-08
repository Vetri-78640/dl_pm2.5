"""Dry run of the final-round notebook (kaggle/final/final_source.py) on small
synthetic data (CPU). Checks code paths, not numbers: the input pre-check stops
the run when an input is missing; seed 1337 trains; the final round reproduces
the logged val scores, the V4 per-seed test predictions and V4's calibration
scales from a fake V4 output; preds.npy, preds_2seed.npy and summary.json are
written. No competition data is used.

Usage: python build_final.py && python dry_run_final.py
"""
import os, sys, json, tempfile, importlib.util
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
T, NT, H, W = 160, 4, 140, 124

root = tempfile.mkdtemp()
os.chdir(root)
spec = importlib.util.spec_from_file_location('final', os.path.join(HERE, '..', 'notebooks', 'kaggle', 'final', 'final_source.py'))
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
C = S.C
rng = np.random.default_rng(0)


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

v4 = os.path.join(root, 'v4_output')
r1 = os.path.join(root, 'round1_resgru_output')
os.makedirs(v4); os.makedirs(r1)
for m in C.MONTHS:
    np.savez_compressed(os.path.join(v4, f'ep_mask_{m}.npz'),
                        mask=(rng.random((T, H, W)) < 0.03).astype(np.uint8),
                        sigma=np.ones((H, W), np.float32))

C.DATA_ROOT, C.RAW, C.TEST = data, os.path.join(data, 'raw'), os.path.join(data, 'test_in')
C.V4_DIR, C.N_TEST, C.A_EPOCHS, C.WORKERS = v4, NT, 2, 0


class Small(S.PM25Windows):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        if self.split == 'train':
            self.samples, self.density = self.samples[:8], self.density[:8]


S.PM25Windows = Small

# fake V4 output (2 seeds, own preds, own calibration) and round-one ResGRU-UNet seed 42
C.WORK = os.path.join(root, 'v4_build')
os.makedirs(C.WORK)
stats = S.build_stats()
os.replace(os.path.join(C.WORK, 'stats.npz'), os.path.join(v4, 'stats.npz'))
months, static = S.load_months(stats)
val = Small(months, static, stats, 'val')
test = S.PM25Test(stats, static)


def fake(make, seed, path):
    torch.manual_seed(seed)
    model = make()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.01 * torch.randn_like(p))
    score = S.evaluate(model, torch.utils.data.DataLoader(val, batch_size=4))['score']
    torch.save({'model': model.state_dict(), 'metrics': {'score': score}, 'epoch': 1}, path)
    return model


vps = []
for seed in (42, 1337):
    m = fake(S.AFNOv4, seed, os.path.join(v4, f's{seed}_phaseA_best.pt'))
    np.save(os.path.join(v4, f'preds_s{seed}.npy'), S.predict(m, test, 4))
    vps.append(S.val_arrays(m, val)[0])
_, vt, vm = S.val_arrays(m, val)
np.save(os.path.join(v4, 'horizon_scales.npy'), S.fit_horizon_scales(np.mean(vps, 0), vt, vm, 'v4'))
fake(lambda: S.ResGRUUNet(C.C_IN), 42, os.path.join(r1, 'resgru_s42_phaseA_best.pt'))
del months, vps

FAIL = []


def check(ok, msg):
    print(('PASS ' if ok else 'FAIL ') + msg)
    if not ok:
        FAIL.append(msg)


C.WORK = os.path.join(root, 'work_final')
os.makedirs(C.WORK)
try:
    S.final_precheck()
    stopped = False
except RuntimeError as e:
    stopped = 'ResGRU-UNet seed 42' in str(e)
check(stopped, 'pre-check stops when the round-one ResGRU-UNet output is missing')

S.FR_RESGRU_DIR = r1
S.final_precheck()
S.main()
S.final_round()
summ = json.load(open(os.path.join(C.WORK, 'summary.json')))
fr = summ.get('final_round', {})
check(summ['seed'] == 1337 and os.path.exists(os.path.join(C.WORK, 'resgru_s1337_phaseA_best.pt')),
      'seed 1337 trained and saved under its own name')
check(summ['status'] == 'complete', f'status {summ["status"]}')
check(fr.get('all_checks_pass') is True, f'final-round checks {fr.get("checks")}')
check(set(fr) >= {'resgru', 'afnov4', 'val_gap_two_seed'}, 'final_round results in summary.json')
for f in ('preds.npy', 'preds_2seed.npy'):
    p = np.load(os.path.join(C.WORK, f))
    check(p.shape == (NT, H, W, 16) and np.isfinite(p).all(), f'{f} {p.shape}, finite')
p, p2 = np.load(os.path.join(C.WORK, 'preds.npy')), np.load(os.path.join(C.WORK, 'preds_2seed.npy'))
sc = np.load(os.path.join(C.WORK, 'horizon_scales_resgru.npy'))
check(np.allclose(p, p2 * sc[None, None, None, :], rtol=1e-6), 'preds.npy = preds_2seed.npy x ResGRU-UNet scales')

print('=' * 70)
print('DRY RUN PASSED' if not FAIL else f'{len(FAIL)} CHECK(S) FAILED: {FAIL}')
sys.exit(1 if FAIL else 0)
