"""Submit the 5-seed extension files in priority order; stop at the first failure.

usage: python3 -I submit_5seed.py <imvetri0/backbone-study folder> <work folder>
Each file is hard-linked (no copy) as <work>/<name>/preds.npy, the name Kaggle expects.
"""
import os, subprocess, sys

B, W = sys.argv[1], sys.argv[2]
COMP = 'anrf-aise-hack-phase-2-theme-2-pollution-forecasting-iitd'
K = ['uvx', '--python', '3.12', '--from', 'kaggle==2.2.5', '--with', 'six', 'kaggle']
SUBS = [
    ('resgru_5seed_cal', 'analysis/preds_resgru_5seed_cal.npy', 'backbone study 5 seeds: ResGRU-UNet 5-seed mean + V4 per-horizon calibration, val 0.9215 (in sample)'),
    ('afnov4_5seed_cal', 'analysis/preds_afnov4_5seed_cal.npy', 'backbone study 5 seeds: AFNO V4 5-seed mean + V4 per-horizon calibration, val 0.9158 (in sample)'),
    ('resgru_5seed', 'analysis/preds_resgru_5seed.npy', 'backbone study 5 seeds: ResGRU-UNet 5-seed mean, no calibration, val 0.9193'),
    ('afnov4_5seed', 'analysis/preds_afnov4_5seed.npy', 'backbone study 5 seeds: AFNO V4 5-seed mean, no calibration, val 0.9145'),
    ('resgru_s1337', 'analysis/preds_resgru_s1337.npy', 'backbone study 5 seeds: ResGRU-UNet seed 1337 only, no calibration, val 0.9152'),
    ('afnov4_s1', 'seeds_afnov4/preds_s1.npy', 'backbone study 5 seeds: AFNO V4 seed 1 only, no calibration, val 0.9118'),
    ('resgru_s1', 'seeds_resgru/preds_s1.npy', 'backbone study 5 seeds: ResGRU-UNet seed 1 only, no calibration, val 0.9169'),
    ('afnov4_s2', 'seeds_afnov4/preds_s2.npy', 'backbone study 5 seeds: AFNO V4 seed 2 only, no calibration, val 0.9123'),
    ('resgru_s2', 'seeds_resgru/preds_s2.npy', 'backbone study 5 seeds: ResGRU-UNet seed 2 only, no calibration, val 0.9178'),
    ('afnov4_s3', 'seeds_afnov4/preds_s3.npy', 'backbone study 5 seeds: AFNO V4 seed 3 only, no calibration, val 0.9108'),
    ('resgru_s3', 'seeds_resgru/preds_s3.npy', 'backbone study 5 seeds: ResGRU-UNet seed 3 only, no calibration, val 0.9158'),
]
done = set(open(os.path.join(W, 'submitted.txt')).read().split()) if os.path.exists(os.path.join(W, 'submitted.txt')) else set()
for name, src, msg in SUBS:
    if name in done:
        print(f'{name}: already submitted, skipped')
        continue
    d = os.path.join(W, name)
    os.makedirs(d, exist_ok=True)
    dst = os.path.join(d, 'preds.npy')
    if os.path.exists(dst) and not os.path.samefile(dst, os.path.join(B, src)):
        print(f'{name}: {dst} exists and is not {src}; stopped')
        break
    if not os.path.exists(dst):
        os.link(os.path.join(B, src), dst)
    r = subprocess.run(K + ['competitions', 'submit', '-c', COMP, '-f', dst, '-m', msg], capture_output=True, text=True)
    print(f'{name}: exit {r.returncode} | {r.stdout.strip()[-160:]} | {r.stderr.strip()[-300:]}')
    if r.returncode != 0 or 'Successfully' not in r.stdout:
        print('stopped at', name)
        break
    with open(os.path.join(W, 'submitted.txt'), 'a') as f:
        f.write(name + '\n')
