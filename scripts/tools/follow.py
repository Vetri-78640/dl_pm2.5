"""Save a Kaggle kernel's live log for N seconds, then stop the whole process group.

usage: python3 follow.py <owner/slug> <out_file> <seconds>

Killing only `uvx` leaves its `kaggle kernels logs -f` child running, so the
group is killed instead.
"""
import os, signal, subprocess, sys, time

K = ['uvx', '--python', '3.12', '--from', 'kaggle==2.2.5', '--with', 'six', 'kaggle']
p = subprocess.Popen(K + ['kernels', 'logs', '-f', sys.argv[1]], stdout=open(sys.argv[2], 'w'),
                     stderr=subprocess.DEVNULL, start_new_session=True)
time.sleep(float(sys.argv[3]))
try:
    os.killpg(p.pid, signal.SIGKILL)
except (ProcessLookupError, PermissionError):
    pass
