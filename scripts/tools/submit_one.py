"""Check one prediction file and submit it as preds.npy (hard link, no copy).

usage: python3 -I submit_one.py <preds file> <work folder> <name> <message>
"""
import os, subprocess, sys
import numpy as np

src, work, name, msg = sys.argv[1:5]
a = np.load(src, mmap_mode='r')
ok = a.shape == (218, 140, 124, 16) and a.dtype == np.float32 and bool(np.isfinite(a).all()) and float(a.min()) >= 0
print(f'{name}: {a.shape} {a.dtype} min {float(a.min()):.4f} mean {float(a.mean()):.3f} -> {"ok" if ok else "REFUSED"}')
if not ok:
    sys.exit(1)
d = os.path.join(work, name)
os.makedirs(d, exist_ok=True)
dst = os.path.join(d, 'preds.npy')
if os.path.exists(dst) and not os.path.samefile(dst, src):
    sys.exit(f'refused: {dst} exists and is not {src}; remove it by hand first')
if not os.path.exists(dst):
    os.link(src, dst)
K = ['uvx', '--python', '3.12', '--from', 'kaggle==2.2.5', '--with', 'six', 'kaggle']
r = subprocess.run(K + ['competitions', 'submit', '-c', 'anrf-aise-hack-phase-2-theme-2-pollution-forecasting-iitd',
                        '-f', dst, '-m', msg], capture_output=True, text=True)
print(f'exit {r.returncode} | {r.stdout.strip().splitlines()[0] if r.stdout.strip() else ""} | {r.stderr.strip()[-200:] if r.returncode else ""}')
sys.exit(0 if r.returncode == 0 and 'Successfully' in r.stdout else 1)
