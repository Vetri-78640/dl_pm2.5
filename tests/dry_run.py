"""End-to-end dry run of study.py main() on small synthetic data (CPU).

Checks the notebook code paths, not the numbers: V4 pipeline check, model
build, persistence at init, training with checkpoints and train_log.csv,
gradient accumulation, a forced time-budget stop followed by a resume,
best-epoch evaluation, preds.npy and summary.json. No competition data is used.

Usage: python dry_run.py [backbone ...]
"""
import os, sys, json, csv, tempfile, importlib.util
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
BBS = sys.argv[1:] or ['convlstm', 'fmgwaclu', 'resgru', 'patchafno']
T, NT, H, W = 160, 4, 140, 124

root = tempfile.mkdtemp()
os.chdir(root)
spec = importlib.util.spec_from_file_location('study', os.path.join(HERE, '..', 'src', 'study.py'))
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
C = S.C

rng = np.random.default_rng(0)
hours = np.arange(T)
diurnal = np.clip(np.sin(2 * np.pi * (hours % 24 - 6) / 24), 0, None)


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
os.makedirs(v4)
for m in C.MONTHS:
    np.savez_compressed(os.path.join(v4, f'ep_mask_{m}.npz'),
                        mask=(rng.random((T, H, W)) < 0.03).astype(np.uint8),
                        sigma=np.ones((H, W), np.float32))

C.DATA_ROOT, C.RAW, C.TEST = data, os.path.join(data, 'raw'), os.path.join(data, 'test_in')
C.V4_DIR, C.N_TEST, C.A_EPOCHS, C.WORKERS = v4, NT, 2, 0

Full = S.PM25Windows


class Small(Full):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        if self.split == 'train':
            self.samples, self.density = self.samples[:8], self.density[:8]


S.PM25Windows = Small

# fake V4 checkpoint: AFNOv4 at init (= persistence), scored with this pipeline
C.WORK = os.path.join(root, 'v4_build')
os.makedirs(C.WORK)
stats = S.build_stats()
os.replace(os.path.join(C.WORK, 'stats.npz'), os.path.join(v4, 'stats.npz'))
months, static = S.load_months(stats)
val = Small(months, static, stats, 'val')
torch.manual_seed(0)
ref = S.AFNOv4()
score = S.evaluate(ref, torch.utils.data.DataLoader(val, batch_size=4))['score']
torch.save({'model': ref.state_dict(), 'metrics': {'score': score}},
           os.path.join(v4, 's42_phaseA_best.pt'))
del months, ref

FAIL = []


def check(ok, msg):
    print(('PASS ' if ok else 'FAIL ') + msg)
    if not ok:
        FAIL.append(msg)


def outputs(work, name):
    summ = json.load(open(os.path.join(work, 'summary.json')))
    rows = list(csv.DictReader(open(os.path.join(work, 'train_log.csv'))))
    p = np.load(os.path.join(work, 'preds.npy'))
    return summ, rows, p


real_pick, real_time = S.pick_micro, S.time_left
for bb in BBS:
    print('#' * 70, '\n#', bb, '\n' + '#' * 70)
    S.BACKBONE = bb
    C.WORK = os.path.join(root, f'work_{bb}')
    os.makedirs(C.WORK)
    name = f'{bb}_s42_phaseA'
    resume_test = bb == BBS[0]
    if resume_test:
        S.pick_micro = lambda model, ds, batch: 2
        calls = {'n': 0}

        def stop_at_epoch_2():
            calls['n'] += 1
            return 1e9 if calls['n'] < 2 else 0.0

        S.time_left = stop_at_epoch_2
        S.main()
        summ, rows, _ = outputs(C.WORK, name)
        check(summ['status'].startswith('incomplete'), f'{bb}: time stop -> status "{summ["status"][:40]}..."')
        check(len(rows) == 1 and not os.path.exists(os.path.join(C.WORK, f'{name}_done.flag')),
              f'{bb}: 1 epoch logged, no done flag after the stop')
        check(summ['accum'] == 2, f'{bb}: micro-batch 2 x accumulation 2')
        S.time_left = real_time
    S.main()
    summ, rows, p = outputs(C.WORK, name)
    check(summ['status'] == 'complete', f'{bb}: status {summ["status"]}')
    check(abs(summ['pipeline_check']['diff']) < 1e-6, f'{bb}: pipeline check diff {summ["pipeline_check"]["diff"]}')
    check([int(r['epoch']) for r in rows] == [1, 2], f'{bb}: train_log epochs {[r["epoch"] for r in rows]}')
    check(p.shape == (NT, H, W, 16) and np.isfinite(p).all(), f'{bb}: preds.npy {p.shape}, finite')
    check(len(summ['best']['per_hour']['gsmape']) == 16, f'{bb}: per-hour metrics for 16 hours')
    check(os.path.exists(os.path.join(C.WORK, f'{name}_done.flag')), f'{bb}: done flag written')
    S.pick_micro = real_pick

print('=' * 70)
print('DRY RUN PASSED' if not FAIL else f'{len(FAIL)} CHECK(S) FAILED: {FAIL}')
sys.exit(1 if FAIL else 0)
