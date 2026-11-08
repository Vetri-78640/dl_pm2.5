# %% [markdown]
# # Code4CleanAir - controlled backbone study: {{TITLE}}
#
# {{SUMMARY}}
#
# **Why this notebook exists.** The team's model families (ConvLSTM, FMG-WACLU,
# ResGRU-UNet, patch AFNO, AFNO V4) were built at different times with
# different pipelines: split, episode masks, input channels, loss and
# post-processing all changed along the way. Their leaderboard scores therefore
# mix architecture with pipeline. This study trains each network inside one
# fixed pipeline (the AFNO V4 pipeline), so the only difference between
# notebooks is the network.
#
# **Protocol (identical in every notebook of the study)**
#
# | item | setting |
# |---|---|
# | inputs | V4's 20 channels: 9 met + wind speed + ventilation + advection + PM tendency + hour sin/cos + diurnal anomaly + 4 static |
# | split | tail 18% of each month is validation, 12 h gap before it |
# | episode masks | official robust STL masks, reused from the V4 run |
# | validation | the same 214 windows, exact competition metric, equal weights |
# | output | log1p(last frame) + predicted change, final layer zero-initialized |
# | training | V4 phase A: composite loss, horizon weights 0.7 -> 1.3, episode weight 6, AdamW 2.5e-4, one-cycle, 16 epochs, effective batch 4, EMA 0.999, AMP, grad clip 1.0 |
# | selection | best epoch by validation score |
# | seed | 42, one run |
# | post-processing | none (no calibration, ensemble, TTA or blend) |
#
# **Reference (not retrained):** AFNO V4 seed 42 phase A best, val 0.9110,
# private 0.8762 without calibration. Persistence: val 0.8515.
#
# **Decision rule (fixed before training):** AFNO V4 is called better than
# this network only if its val score is higher by more than 0.003 and its
# private score is also higher. A val gap of 0.003 or less is a tie.
#
# **Safety:** the run stops early if no usable GPU is present, if the data or
# the V4 output is missing, or if re-evaluating the V4 checkpoint does not
# reproduce its logged val score (0.9110 +/- 0.002). Checkpoints are saved
# every epoch, `train_log.csv` gets one row per epoch, and `preds.npy` plus
# `summary.json` are written at the end.
#
# {{MODEL_NOTES}}

# %%
import os, gc, math, time, json, shutil, traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import multiprocessing as mp

try:
    MP_CTX = mp.get_context('fork')
except ValueError:
    MP_CTX = mp.get_context()

T0_GLOBAL = time.time()
BACKBONE = 'resgru'   # convlstm | fmgwaclu | resgru | patchafno
SEED = 42             # 42 in round one; 1337 in the final round


def find_dir(root, must, depth=6):
    """First directory under root that contains every relative path in must."""
    if not os.path.isdir(root):
        return None
    base = root.rstrip(os.sep).count(os.sep)
    for dp, dn, _ in os.walk(root):
        if all(os.path.exists(os.path.join(dp, m)) for m in must):
            return dp
        if dp.count(os.sep) - base >= depth:
            dn[:] = []
    return None


class C:
    IS_KAGGLE = os.path.exists('/kaggle/input')
    DATA_ROOT = ((find_dir('/kaggle/input', ['raw/APRIL_16/cpm25.npy', 'test_in/cpm25.npy']) or '')
                 if IS_KAGGLE else 'Dataset')
    # V4 run output, attached as a notebook source: masks, stats, seed-42 checkpoint
    V4_DIR = (find_dir('/kaggle/input', ['ep_mask_APRIL_16.npz', 'stats.npz', 's42_phaseA_best.pt'])
              if IS_KAGGLE else None)
    # this notebook's own earlier output, if attached, to resume an interrupted run
    RESUME_DIR = (find_dir('/kaggle/input', [f'{BACKBONE}_s{SEED}_phaseA_last.pt'])
                  if IS_KAGGLE else None)
    WORK = '/kaggle/working' if IS_KAGGLE else './work'

    RAW = os.path.join(DATA_ROOT, 'raw')
    TEST = os.path.join(DATA_ROOT, 'test_in')

    MONTHS = ['APRIL_16', 'JULY_16', 'OCT_16', 'DEC_16']
    FEATS = ['cpm25', 'q2', 't2', 'u10', 'v10', 'pblh', 'psfc', 'swdown', 'rain']
    LOG_FEATS = {'cpm25', 'rain'}
    N_PRE = 12
    H, W = 140, 124
    T_IN, T_OUT = 10, 16
    WIN = 26
    N_TEST = 218
    C_IN = 20
    U_IDX, V_IDX = 3, 4    # u10, v10 positions in the 20-channel input

    VAL_FRAC = 0.18
    GAP = 12
    VAL_STRIDE = 2
    STL_ROBUST = True
    STL_PROCS = 4

    # AFNO V4 network (used for the pipeline check)
    ENC = [64, 128, 256]
    AFNO_BLOCKS = 6
    MID_AFNO_BLOCKS = 1
    POST_AFNO_BLOCKS = 2
    AFNO_HEADS = 8
    MLP_RATIO = 2.0
    DROP = 0.1
    TRANS_LAYERS = 4
    CLIP_IN = 10.0

    # V4 phase A recipe, unchanged
    A_EPOCHS, A_LR, A_BATCH, A_PATIENCE = 16, 2.5e-4, 4, 16
    EP_WT_A = 6.0
    LOSS_A = dict(huber=0.40, gsmape=0.25, esmape=0.15, ecorr=0.20)
    HW_MIN, HW_MAX = 0.7, 1.3
    EMA_DECAY = 0.999
    WD = 1e-4

    SEED = SEED
    WORKERS = 2
    MAX_HOURS = 10.5
    CHECK_TOL = 0.002
    V4_REF = dict(val=0.9110, private=0.8762, epoch=4)


def seed_all(seed=C.SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def art(name):
    for d in (C.WORK, C.RESUME_DIR, C.V4_DIR):
        if d:
            p = os.path.join(d, name)
            if os.path.exists(p):
                return p
    return None


os.makedirs(C.WORK, exist_ok=True)
seed_all()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def check_gpu():
    """Fail fast instead of silently training on CPU (CAMS v2 lost 12 h that way)."""
    if not torch.cuda.is_available():
        raise RuntimeError('No CUDA GPU. Select the T4 accelerator; CPU training '
                           'cannot finish inside a Kaggle session.')
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    try:
        a = torch.randn(256, 256, device='cuda')
        (a @ a).sum().item()
        nn.Conv2d(4, 4, 3).cuda()(torch.randn(1, 4, 16, 16, device='cuda')).sum().item()
    except Exception as e:
        raise RuntimeError(f'GPU {name} (sm_{cap[0]}{cap[1]}) is not usable with '
                           f'torch {torch.__version__}: {e}')
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'GPU: {name}  sm_{cap[0]}{cap[1]}  {mem:.1f} GB  torch {torch.__version__}')
    return name


print('device:', DEVICE, ' backbone:', BACKBONE)

# %% [markdown]
# ## 1. Official episode masks (robust STL, period 24) - reused from the V4 run
#
# The masks are loaded from the attached V4 output, so every model is scored
# on exactly the same episode pixels. The STL code below only runs if a mask
# file is missing.

# %%
_STL_DATA = None

def _stl_init(data_2d):
    global _STL_DATA
    _STL_DATA = data_2d

def _stl_chunk(args):
    from statsmodels.tsa.seasonal import STL
    lo, hi, robust = args
    T = _STL_DATA.shape[0]
    mask = np.zeros((T, hi - lo), dtype=np.uint8)
    sig = np.zeros(hi - lo, dtype=np.float32)
    for j, k in enumerate(range(lo, hi)):
        ts = _STL_DATA[:, k].astype(np.float64)
        res = STL(ts, period=24, robust=robust).fit()
        r = res.resid
        s = np.std(r) + 1e-8
        sig[j] = s
        mask[:, j] = ((r > 3 * s) & (ts > 1)).astype(np.uint8)
    return lo, hi, mask, sig


def episode_masks(month):
    cached = art(f'ep_mask_{month}.npz')
    if cached:
        d = np.load(cached)
        return d['mask'], d['sigma']
    print(f'  WARNING: no cached mask for {month}, computing STL (about 14 min)...')
    data = np.load(os.path.join(C.RAW, month, 'cpm25.npy')).astype(np.float32)
    T = data.shape[0]
    data_2d = data.reshape(T, -1)
    P = data_2d.shape[1]
    step = 512
    jobs = [(lo, min(lo + step, P), C.STL_ROBUST) for lo in range(0, P, step)]
    mask_2d = np.zeros((T, P), dtype=np.uint8)
    sigma = np.zeros(P, dtype=np.float32)
    t0 = time.time()
    with MP_CTX.Pool(C.STL_PROCS, initializer=_stl_init,
                     initargs=(data_2d,)) as pool:
        it = pool.imap_unordered(_stl_chunk, jobs)
        for i in range(len(jobs)):
            lo, hi, m, s = it.next(timeout=600)
            mask_2d[:, lo:hi] = m
            sigma[lo:hi] = s
            if i % 8 == 0:
                print(f'    {month}: {i+1}/{len(jobs)} chunks  {time.time()-t0:.0f}s')
    mask = mask_2d.reshape(T, C.H, C.W)
    sigma = sigma.reshape(C.H, C.W)
    np.savez_compressed(os.path.join(C.WORK, f'ep_mask_{month}.npz'),
                        mask=mask, sigma=sigma)
    print(f'  {month}: episode ratio={mask.mean():.4f}  ({time.time()-t0:.0f}s)')
    return mask, sigma


# %% [markdown]
# ## 2. Normalization stats and physics-derived channels (V4, unchanged)
#
# Per-pixel mean/std over the train region for 12 dynamic channels (9 met +
# wind speed + ventilation log1p(pblh x ws) + advection of log PM2.5), the
# per-pixel hour-of-day climatology, the PM tendency std, and the diurnal
# templates used to estimate the start hour of a window. Loaded from the V4
# output when present.

# %%
def month_val_start(T):
    return T - int(T * C.VAL_FRAC)


def _derived(arrs):
    """ws, vent, adv from raw arrays. Works for (T,H,W) with any T."""
    ws = np.sqrt(arrs['u10'] ** 2 + arrs['v10'] ** 2)
    vent = np.log1p(arrs['pblh'] * ws)
    logc = np.log1p(arrs['cpm25'])
    gy = np.gradient(logc, axis=1)
    gx = np.gradient(logc, axis=2)
    adv = -(arrs['u10'] * gx + arrs['v10'] * gy)
    return ws, vent, adv


def build_stats():
    cached = art('stats.npz')
    if cached:
        d = np.load(cached)
        return {k: d[k] for k in d.files}
    print('computing normalization stats (train region only)...')
    nf = C.N_PRE
    s = np.zeros((nf, C.H, C.W), np.float64)
    sq = np.zeros((nf, C.H, C.W), np.float64)
    n = 0
    clim24 = np.zeros((24, C.H, C.W), np.float64)
    clim_cnt = np.zeros(24)
    dpm_s = np.zeros((C.H, C.W), np.float64)
    dpm_sq = np.zeros((C.H, C.W), np.float64)
    dpm_n = 0
    sw_prof = np.zeros(24); t2_prof = np.zeros(24); cnt = np.zeros(24)
    for month in C.MONTHS:
        arrs = {}
        vb = None
        for f in C.FEATS:
            a = np.load(os.path.join(C.RAW, month, f'{f}.npy')).astype(np.float32)
            if vb is None:
                vb = month_val_start(a.shape[0])
            arrs[f] = a
            x = a[:vb]
            if f in C.LOG_FEATS:
                x = np.log1p(x)
            i = C.FEATS.index(f)
            s[i] += x.sum(0); sq[i] += (x.astype(np.float64) ** 2).sum(0)
        tr = {k: v[:vb] for k, v in arrs.items()}
        ws, vent, adv = _derived(tr)
        for i, x in zip(range(9, 12), (ws, vent, adv)):
            s[i] += x.sum(0); sq[i] += (x.astype(np.float64) ** 2).sum(0)
        n += vb
        logc = np.log1p(tr['cpm25'])
        for h in range(24):
            sel = logc[h::24]
            clim24[h] += sel.sum(0); clim_cnt[h] += sel.shape[0]
        d = np.diff(logc, axis=0)
        dpm_s += d.sum(0); dpm_sq += (d.astype(np.float64) ** 2).sum(0)
        dpm_n += d.shape[0]
        sm = tr['swdown'].mean(axis=(1, 2))
        tm = tr['t2'].mean(axis=(1, 2))
        for i in range(len(sm)):
            h = i % 24
            sw_prof[h] += sm[i]; t2_prof[h] += tm[i]; cnt[h] += 1
        del arrs, tr; gc.collect()
    mean = (s / n).astype(np.float32)
    std = np.sqrt(np.maximum(sq / n - (s / n) ** 2, 1e-12)).astype(np.float32)
    med = np.median(std.reshape(nf, -1), axis=1)
    std = np.maximum(std, 0.1 * med[:, None, None])
    dpm_std = np.sqrt(np.maximum(dpm_sq / dpm_n - (dpm_s / dpm_n) ** 2,
                                 1e-12)).astype(np.float32)
    dpm_std = np.maximum(dpm_std, 0.1 * np.median(dpm_std))
    out = dict(mean=mean, std=std,
               clim24=(clim24 / clim_cnt[:, None, None]).astype(np.float32),
               dpm_std=dpm_std,
               sw_prof=(sw_prof / cnt).astype(np.float32),
               t2_prof=(t2_prof / cnt).astype(np.float32))
    ll = np.load(os.path.join(C.RAW, 'lat_long.npy')).astype(np.float32)
    out['lat'] = ll[..., 0]; out['lon'] = ll[..., 1]
    np.savez(os.path.join(C.WORK, 'stats.npz'), **out)
    return out


def normalize_feats(arrs, stats):
    """arrs: dict feat -> (T,H,W) raw. Returns (T, 12, H, W) float16."""
    T = arrs['cpm25'].shape[0]
    cl = C.CLIP_IN
    out = np.empty((T, C.N_PRE, C.H, C.W), np.float16)
    for i, f in enumerate(C.FEATS):
        x = arrs[f]
        if f in C.LOG_FEATS:
            x = np.log1p(x)
        z = (x - stats['mean'][i]) / stats['std'][i]
        out[:, i] = np.clip(z, -cl, cl).astype(np.float16)
    for i, x in zip(range(9, 12), _derived(arrs)):
        z = (x - stats['mean'][i]) / stats['std'][i]
        out[:, i] = np.clip(z, -cl, cl).astype(np.float16)
    return out


def build_static(stats, sigma_mean):
    lat, lon = stats['lat'], stats['lon']
    latn = (lat - lat.min()) / (lat.max() - lat.min()) * 2 - 1
    lonn = (lon - lon.min()) / (lon.max() - lon.min()) * 2 - 1
    clim = stats['mean'][0]
    climn = (clim - clim.mean()) / (clim.std() + 1e-6)
    sig = np.log1p(sigma_mean)
    sign = (sig - sig.mean()) / (sig.std() + 1e-6)
    return np.stack([latn, lonn, climn, sign]).astype(np.float32)


def load_months(stats):
    months = {}
    sigmas = []
    for month in C.MONTHS:
        mask, sigma = episode_masks(month)
        sigmas.append(sigma)
        arrs = {f: np.load(os.path.join(C.RAW, month, f'{f}.npy')).astype(np.float32)
                for f in C.FEATS}
        months[month] = dict(
            feats=normalize_feats(arrs, stats),
            raw=arrs['cpm25'],
            log=np.log1p(arrs['cpm25']),
            mask=mask,
            T=arrs['cpm25'].shape[0],
            sw_mean=arrs['swdown'].mean(axis=(1, 2)),
            t2_mean=arrs['t2'].mean(axis=(1, 2)),
        )
        del arrs; gc.collect()
        print(f'  {month}: T={months[month]["T"]}  ep_ratio={mask.mean():.4f}')
    static = build_static(stats, np.mean(sigmas, axis=0))
    return months, static


# %% [markdown]
# ## 3. Datasets - leak-free split, window-local channels (V4, unchanged)
#
# Train windows end 12 h before the validation block of each month.
# Validation windows use the estimated start hour, exactly as the test
# windows do, so validation sees the same hour-feature noise as the test.

# %%
def hour_channels(start_hour):
    h = (start_hour + np.arange(C.T_IN)) % 24
    hs = np.sin(2 * np.pi * h / 24).astype(np.float32)
    hc = np.cos(2 * np.pi * h / 24).astype(np.float32)
    return hs, hc


def window_channels(logw, h0, stats):
    """dpm + clim_anom for a (T_IN,H,W) log1p-cpm25 window starting at h0."""
    cl = C.CLIP_IN
    dpm = np.diff(logw, axis=0, prepend=logw[:1]) / stats['dpm_std']
    hrs = (h0 + np.arange(C.T_IN)) % 24
    anom = (logw - stats['clim24'][hrs]) / stats['std'][0]
    return (np.clip(dpm, -cl, cl).astype(np.float32),
            np.clip(anom, -cl, cl).astype(np.float32))


def estimate_start_hour(sw_seq, t2_seq, sw_prof, t2_prof):
    best, bh = 1e18, 0
    t2c = t2_seq - t2_seq.mean()
    for h0 in range(24):
        idx = (h0 + np.arange(C.T_IN)) % 24
        e_sw = ((sw_seq - sw_prof[idx]) ** 2).mean() / (sw_prof.max() ** 2)
        tpl = t2_prof[idx] - t2_prof[idx].mean()
        e_t2 = ((t2c - tpl) ** 2).mean() / max(t2_prof.std() ** 2, 1e-6)
        e = 24 * e_sw + e_t2
        if e < best:
            best, bh = e, h0
    return bh


class PM25Windows(Dataset):
    def __init__(self, months, static, stats, split):
        self.months = months
        self.static = static
        self.stats = stats
        self.split = split
        self.keys = list(months.keys())
        self.samples = []
        for mi, k in enumerate(self.keys):
            T = months[k]['T']
            vb = month_val_start(T)
            if split == 'train':
                starts = range(0, vb - C.WIN - C.GAP + 1)
            else:
                starts = range(vb, T - C.WIN + 1, C.VAL_STRIDE)
            self.samples += [(mi, s) for s in starts]
        dens = []
        for mi, s in self.samples:
            m = self.months[self.keys[mi]]['mask']
            dens.append(m[s + C.T_IN:s + C.WIN].mean())
        self.density = np.array(dens, np.float32)
        self.est_hours = None
        if split != 'train':
            self.est_hours = np.array([
                estimate_start_hour(
                    months[self.keys[mi]]['sw_mean'][s:s + C.T_IN],
                    months[self.keys[mi]]['t2_mean'][s:s + C.T_IN],
                    stats['sw_prof'], stats['t2_prof'])
                for mi, s in self.samples])
            true_h = np.array([s % 24 for _, s in self.samples])
            err = np.minimum((self.est_hours - true_h) % 24,
                             (true_h - self.est_hours) % 24)
            print(f'  val hour estimation: <=1h {np.mean(err <= 1):.3f}')
        print(f'  {split}: {len(self.samples)} windows  '
              f'ep_density mean={self.density.mean():.4f}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        mi, s = self.samples[i]
        m = self.months[self.keys[mi]]
        h0 = int(self.est_hours[i]) if self.est_hours is not None else s % 24
        x = np.empty((C.C_IN, C.T_IN, C.H, C.W), np.float32)
        x[:C.N_PRE] = m['feats'][s:s + C.T_IN].astype(np.float32).transpose(1, 0, 2, 3)
        logw = m['log'][s:s + C.T_IN]
        dpm, anom = window_channels(logw, h0, self.stats)
        x[12] = dpm
        hs, hc = hour_channels(h0)
        x[13] = hs[:, None, None]
        x[14] = hc[:, None, None]
        x[15] = anom
        x[16:20] = self.static[:, None]
        t_log = m['log'][s + C.T_IN:s + C.WIN]
        t_raw = m['raw'][s + C.T_IN:s + C.WIN]
        last_log = m['log'][s + C.T_IN - 1]
        mask = m['mask'][s + C.T_IN:s + C.WIN].astype(np.float32)
        return (torch.from_numpy(x), torch.from_numpy(t_log.copy()),
                torch.from_numpy(t_raw.copy()), torch.from_numpy(last_log.copy()),
                torch.from_numpy(mask))


class PM25Test(Dataset):
    def __init__(self, stats, static):
        self.static = static
        self.stats = stats
        self.arrs = {f: np.load(os.path.join(C.TEST, f'{f}.npy'), mmap_mode='r')
                     for f in C.FEATS}
        sw = np.asarray(self.arrs['swdown'], np.float32).mean(axis=(2, 3))
        t2 = np.asarray(self.arrs['t2'], np.float32).mean(axis=(2, 3))
        self.hours = np.array([
            estimate_start_hour(sw[i], t2[i], stats['sw_prof'], stats['t2_prof'])
            for i in range(sw.shape[0])])
        print('  test start-hour histogram:',
              np.bincount(self.hours, minlength=24).tolist())

    def __len__(self):
        return self.arrs['cpm25'].shape[0]

    def __getitem__(self, i):
        raw = {f: np.asarray(self.arrs[f][i], np.float32) for f in C.FEATS}
        feats = normalize_feats(raw, self.stats).astype(np.float32)
        h0 = int(self.hours[i])
        x = np.empty((C.C_IN, C.T_IN, C.H, C.W), np.float32)
        x[:C.N_PRE] = feats.transpose(1, 0, 2, 3)
        logw = np.log1p(raw['cpm25'])
        dpm, anom = window_channels(logw, h0, self.stats)
        x[12] = dpm
        hs, hc = hour_channels(h0)
        x[13] = hs[:, None, None]
        x[14] = hc[:, None, None]
        x[15] = anom
        x[16:20] = self.static[:, None]
        last_log = logw[-1]
        return torch.from_numpy(x), torch.from_numpy(last_log.copy())


