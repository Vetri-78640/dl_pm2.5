"""Check the 11 five-seed-extension submission files before upload.

usage: python3 -I check_subs.py <imvetri0/backbone-study folder>
Checks: same shape and dtype, finite, non-negative, all files distinct, and each
5-seed ensemble agrees closely with the mean of its 4 local seeds.
"""
import hashlib, os, sys
import numpy as np

B = sys.argv[1]
F = {
    'resgru_s1337': 'analysis/preds_resgru_s1337.npy',
    'resgru_s1': 'seeds_resgru/preds_s1.npy', 'resgru_s2': 'seeds_resgru/preds_s2.npy', 'resgru_s3': 'seeds_resgru/preds_s3.npy',
    'afnov4_s1': 'seeds_afnov4/preds_s1.npy', 'afnov4_s2': 'seeds_afnov4/preds_s2.npy', 'afnov4_s3': 'seeds_afnov4/preds_s3.npy',
    'resgru_5seed': 'analysis/preds_resgru_5seed.npy', 'resgru_5seed_cal': 'analysis/preds_resgru_5seed_cal.npy',
    'afnov4_5seed': 'analysis/preds_afnov4_5seed.npy', 'afnov4_5seed_cal': 'analysis/preds_afnov4_5seed_cal.npy',
}
P, ok, digests = {}, True, {}
for k, f in F.items():
    a = np.load(os.path.join(B, f), mmap_mode='r')
    fin = bool(np.isfinite(a).all())
    mn = float(a.min())
    h = hashlib.sha256(open(os.path.join(B, f), 'rb').read()).hexdigest()[:16]
    digests.setdefault(h, []).append(k)
    print(f'{k:18s} {a.shape} {a.dtype} finite {fin} min {mn:.4f} mean {float(a.mean()):.3f} sha {h}')
    ok &= fin and mn >= 0 and a.shape == np.load(os.path.join(B, F['resgru_s1']), mmap_mode='r').shape
    P[k] = a
dup = [v for v in digests.values() if len(v) > 1]
print('duplicates:', dup or 'none')
ok &= not dup
for fam, seeds in (('resgru', ['s1337', 's1', 's2', 's3']), ('afnov4', ['s1', 's2', 's3'])):
    m = np.mean([np.asarray(P[f'{fam}_{s}'], dtype=np.float64) for s in seeds], axis=0)
    e = np.asarray(P[f'{fam}_5seed'], dtype=np.float64)
    c = np.corrcoef(np.log1p(m).ravel(), np.log1p(e).ravel())[0, 1]
    r = float(np.mean(e) / np.mean(m))
    print(f'{fam}: 5-seed ensemble vs mean of {len(seeds)} local seeds: log1p corr {c:.5f}, mean ratio {r:.4f}')
    ok &= c > 0.99
    cal = np.asarray(P[f'{fam}_5seed_cal'], dtype=np.float64)
    ratio = (cal.mean(axis=(0, 1, 2)) / e.mean(axis=(0, 1, 2)))
    print(f'{fam}: calibrated / uncalibrated mean per lead hour: {np.round(ratio, 3).tolist()}')
print('ALL OK' if ok else 'PROBLEM FOUND')
