"""Dry run of the seed-loop notebooks (kaggle/seeds_<bb>/seeds_source.py) on small
synthetic data (CPU). Checks: both seeds train and save seed-named checkpoints,
summaries and predictions; a second run skips finished seeds and adds no epochs.
No competition data is used.

Usage: python build_seeds.py && python dry_run_seeds.py [resgru afnov4]
"""
import os, sys, json, csv, tempfile, importlib.util
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
BBS = sys.argv[1:] or ['resgru', 'afnov4']
T, NT, H, W = 160, 4, 140, 124
root = tempfile.mkdtemp()
os.chdir(root)
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


def load(bb):
    spec = importlib.util.spec_from_file_location(f'seeds_{bb}', os.path.join(HERE, '..', 'notebooks', 'kaggle', f'seeds_{bb}', 'seeds_source.py'))
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    return S


S0 = load(BBS[0])
C0 = S0.C
data = os.path.join(root, 'data')
for m in C0.MONTHS:
    os.makedirs(os.path.join(data, 'raw', m))
    for f in C0.FEATS:
        np.save(os.path.join(data, 'raw', m, f'{f}.npy'), field(f, T))
os.makedirs(os.path.join(data, 'test_in'))
for f in C0.FEATS:
    np.save(os.path.join(data, 'test_in', f'{f}.npy'),
            np.stack([field(f, C0.T_IN, h0=5 * i) for i in range(NT)]))
ll = np.stack(np.meshgrid(np.linspace(8, 37, H), np.linspace(68, 97, W), indexing='ij'), -1)
np.save(os.path.join(data, 'raw', 'lat_long.npy'), ll.astype(np.float32))
v4 = os.path.join(root, 'v4_output')
os.makedirs(v4)
for m in C0.MONTHS:
    np.savez_compressed(os.path.join(v4, f'ep_mask_{m}.npz'),
                        mask=(rng.random((T, H, W)) < 0.03).astype(np.uint8),
                        sigma=np.ones((H, W), np.float32))


def setup(S, work):
    C = S.C
    C.DATA_ROOT, C.RAW, C.TEST = data, os.path.join(data, 'raw'), os.path.join(data, 'test_in')
    C.V4_DIR, C.N_TEST, C.A_EPOCHS, C.WORKERS, C.WORK = v4, NT, 1, 0, work
    os.makedirs(work, exist_ok=True)
    Full = S.PM25Windows

    class Small(Full):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            if self.split == 'train':
                self.samples, self.density = self.samples[:8], self.density[:8]

    S.PM25Windows = Small
    S.SEEDS = [1, 2]


setup(S0, os.path.join(root, 'v4_build'))
stats = S0.build_stats()
os.replace(os.path.join(C0.WORK, 'stats.npz'), os.path.join(v4, 'stats.npz'))
months, static = S0.load_months(stats)
val = S0.PM25Windows(months, static, stats, 'val')
torch.manual_seed(0)
ref = S0.AFNOv4()
score = S0.evaluate(ref, torch.utils.data.DataLoader(val, batch_size=4))['score']
torch.save({'model': ref.state_dict(), 'metrics': {'score': score}}, os.path.join(v4, 's42_phaseA_best.pt'))
del months, ref

for bb in BBS:
    print('#' * 70, '\n#', bb, '\n' + '#' * 70)
    S = load(bb)
    work = os.path.join(root, f'work_{bb}')
    setup(S, work)
    S.run_seeds()
    names = sorted(os.listdir(work))
    for s in (1, 2):
        need = [f'{bb}_s{s}_phaseA_best.pt', f'{bb}_s{s}_phaseA_done.flag', f'summary_s{s}.json', f'preds_s{s}.npy']
        check(all(n in names for n in need), f'{bb} seed {s}: checkpoint, flag, summary, preds present')
        summ = json.load(open(os.path.join(work, f'summary_s{s}.json')))
        check(summ['seed'] == s and summ['status'] == 'complete', f'{bb} seed {s}: summary seed {summ["seed"]}, status {summ["status"]}')
        p = np.load(os.path.join(work, f'preds_s{s}.npy'))
        check(p.shape == (NT, H, W, 16) and np.isfinite(p).all(), f'{bb} seed {s}: preds {p.shape}, finite')
    rows = list(csv.DictReader(open(os.path.join(work, 'train_log.csv'))))
    check(len(rows) == 2, f'{bb}: 2 epochs logged (1 per seed), got {len(rows)}')
    p1, p2 = np.load(os.path.join(work, 'preds_s1.npy')), np.load(os.path.join(work, 'preds_s2.npy'))
    check(not np.allclose(p1, p2), f'{bb}: seeds 1 and 2 give different predictions')
    S.run_seeds()
    rows = list(csv.DictReader(open(os.path.join(work, 'train_log.csv'))))
    check(len(rows) == 2, f'{bb}: rerun skips finished seeds (still 2 epochs logged)')
    check(np.allclose(np.load(os.path.join(work, 'preds_s1.npy')), p1), f'{bb}: rerun reproduces seed-1 predictions')

print('=' * 70)
print('DRY RUN PASSED' if not FAIL else f'{len(FAIL)} CHECK(S) FAILED: {FAIL}')
sys.exit(1 if FAIL else 0)
