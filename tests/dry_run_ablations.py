"""Dry run of the ablation notebooks (kaggle/ablations_<part>/ablations_source.py) on
small synthetic data (CPU). Checks: each variant changes what it claims to change,
starts as persistence, trains, and writes variant-named outputs; a rerun skips
finished variants. No competition data is used.

Usage: python build_ablations.py && python dry_run_ablations.py
"""
import os, sys, json, csv, tempfile, importlib.util
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
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


def load(part):
    spec = importlib.util.spec_from_file_location(f'abl_{part}', os.path.join(HERE, '..', 'notebooks', 'kaggle', f'ablations_{part}', 'ablations_source.py'))
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    return S


S = load('a')
C = S.C
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
v4 = os.path.join(root, 'v4_output')
os.makedirs(v4)
for m in C.MONTHS:
    np.savez_compressed(os.path.join(v4, f'ep_mask_{m}.npz'),
                        mask=(rng.random((T, H, W)) < 0.03).astype(np.uint8), sigma=np.ones((H, W), np.float32))


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


setup(S, os.path.join(root, 'v4_build'))
stats = S.build_stats()
os.replace(os.path.join(C.WORK, 'stats.npz'), os.path.join(v4, 'stats.npz'))
months, static = S.load_months(stats)
val = S.PM25Windows(months, static, stats, 'val')
torch.manual_seed(0)
ref = S.AFNOv4()
score = S.evaluate(ref, torch.utils.data.DataLoader(val, batch_size=4))['score']
torch.save({'model': ref.state_dict(), 'metrics': {'score': score}}, os.path.join(v4, 's42_phaseA_best.pt'))
del months

n = lambda m: sum(p.numel() for p in m.parameters())
base_a, base_r = n(S.AFNOv4()), n(S.ResGRUUNet(C.C_IN))
pad = S.BACKBONES['afnov4_pad']()
blocks = [b for b in pad.modules() if isinstance(b, S.AFNOBlock)]
check(len(blocks) == 9 and all(type(b.afno) is S.AFNO2DPad for b in blocks), f'pad: all {len(blocks)} AFNO blocks padded')
check(n(pad) == base_a, f'pad: same parameters as AFNO V4 ({n(pad):,})')
ns = S.BACKBONES['afnov4_nosparse']()
check(all(b.afno.sparsity == 0.0 for b in ns.modules() if isinstance(b, S.AFNOBlock)), 'nosparse: soft-threshold 0 in every block')
check(all(b.afno.sparsity == 0.01 for b in S.AFNOv4().modules() if isinstance(b, S.AFNOBlock)), 'base AFNO V4 keeps soft-threshold 0.01')
blk = blocks[0].afno
check(blk(torch.randn(1, blk.nb * blk.bs, 18, 16)).shape == (1, blk.nb * blk.bs, 18, 16), 'pad: output cropped back to 18x16')
al = S.BACKBONES['afnov4_allskip']()
check(al.fuse0.in_channels == C.T_IN * C.ENC[0] and al.fuse2.in_channels == C.T_IN * C.ENC[2], 'allskip: skips take all 10 hours')
nr = S.BACKBONES['resgru_norec']()
check(not hasattr(nr, 'convgru') and hasattr(nr, 'lead_block'), 'norec: ConvGRU removed, lead block added')
check(abs(n(nr) - base_r) / base_r < 1e-3, f'norec: parameters within 0.1% of ResGRU-UNet ({n(nr):,} vs {base_r:,})')
print(f'params: AFNO V4 {base_a:,}, allskip {n(al):,}, pad {n(pad):,}, nosparse {n(ns):,}; ResGRU-UNet {base_r:,}, norec {n(nr):,}')

xb = torch.randn(1, C.C_IN, C.T_IN, H, W)
lb = torch.randn(1, H, W)
with torch.no_grad():
    for bb in ('afnov4_allskip', 'afnov4_pad', 'afnov4_nosparse', 'resgru_norec'):
        m = S.BACKBONES[bb]().eval()
        out = m(xb, lb)
        check(out.shape == (1, 16, H, W) and torch.equal(out, lb.unsqueeze(1).expand_as(out)), f'{bb}: starts as persistence')
    torch.manual_seed(0)
    a = S.AFNOv4().eval()
    torch.manual_seed(0)
    b = S.AFNOv4AllSkip().eval()
    shared = {k: v for k, v in a.state_dict().items() if not k.startswith('fuse')}
    same = all(torch.equal(v, b.state_dict()[k]) for k, v in shared.items())
    torch.nn.init.normal_(a.head[-1].weight, std=0.1)
    b.head[-1].weight.copy_(a.head[-1].weight)
    gap = (a(xb, lb) - b(xb, lb)).abs().max().item()
    check(same and gap < 1e-4, f'allskip: same weights as AFNO V4 outside the skips; same output at init (max diff {gap:.1e})')

for part in ('a', 'b'):
    S = load(part)
    work = os.path.join(root, f'work_{part}')
    setup(S, work)
    S.run_variants()
    for bb in S.VARIANTS:
        names = os.listdir(work)
        need = [f'{bb}_s42_phaseA_best.pt', f'{bb}_s42_phaseA_done.flag', f'summary_{bb}.json', f'preds_{bb}.npy']
        check(all(x in names for x in need), f'{bb}: checkpoint, flag, summary, preds present')
        summ = json.load(open(os.path.join(work, f'summary_{bb}.json')))
        check(summ['backbone'] == bb and summ['status'] == 'complete', f'{bb}: summary backbone {summ["backbone"]}, status {summ["status"]}')
        p = np.load(os.path.join(work, f'preds_{bb}.npy'))
        check(p.shape == (NT, H, W, 16) and np.isfinite(p).all(), f'{bb}: preds {p.shape}, finite')
    rows = len(list(csv.DictReader(open(os.path.join(work, 'train_log.csv')))))
    S.run_variants()
    rows2 = len(list(csv.DictReader(open(os.path.join(work, 'train_log.csv')))))
    check(rows == 2 and rows2 == 2, f'part {part}: 2 epochs logged, rerun adds none ({rows} -> {rows2})')

print('=' * 70)
print('DRY RUN PASSED' if not FAIL else f'{len(FAIL)} CHECK(S) FAILED: {FAIL}')
sys.exit(1 if FAIL else 0)
