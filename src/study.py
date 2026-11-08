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


# %% [markdown]
# ## 4a. Reference network: AFNO V4 (verbatim from `afno_v4.py`)
#
# Not trained here. It is only used to re-evaluate the V4 seed-42 checkpoint
# on this notebook's validation set, which proves that the pipeline in this
# notebook is the one V4 was scored with.

# %%
BACKBONES = {}


class AFNO2D(nn.Module):
    def __init__(self, dim, num_blocks=8, sparsity=0.01):
        super().__init__()
        assert dim % num_blocks == 0
        self.nb = num_blocks
        self.bs = dim // num_blocks
        self.sparsity = sparsity
        sc = 0.02
        self.w1 = nn.Parameter(sc * torch.randn(2, self.nb, self.bs, self.bs))
        self.w2 = nn.Parameter(sc * torch.randn(2, self.nb, self.bs, self.bs))
        self.b1 = nn.Parameter(torch.zeros(2, self.nb, self.bs))
        self.b2 = nn.Parameter(torch.zeros(2, self.nb, self.bs))

    def forward(self, x):
        B, Ch, H, W = x.shape
        res = x
        x = x.permute(0, 2, 3, 1).float()
        xf = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        _, fh, fw, _ = xf.shape
        xf = xf.reshape(B, fh, fw, self.nb, self.bs)
        o1r = F.relu(torch.einsum('bhwni,nio->bhwno', xf.real, self.w1[0])
                     - torch.einsum('bhwni,nio->bhwno', xf.imag, self.w1[1]) + self.b1[0])
        o1i = F.relu(torch.einsum('bhwni,nio->bhwno', xf.imag, self.w1[0])
                     + torch.einsum('bhwni,nio->bhwno', xf.real, self.w1[1]) + self.b1[1])
        o2r = (torch.einsum('bhwni,nio->bhwno', o1r, self.w2[0])
               - torch.einsum('bhwni,nio->bhwno', o1i, self.w2[1]) + self.b2[0])
        o2i = (torch.einsum('bhwni,nio->bhwno', o1i, self.w2[0])
               + torch.einsum('bhwni,nio->bhwno', o1r, self.w2[1]) + self.b2[1])
        o = torch.stack([o2r, o2i], dim=-1)
        o = F.softshrink(o, lambd=self.sparsity)
        o = torch.view_as_complex(o.contiguous()).reshape(B, fh, fw, Ch)
        x = torch.fft.irfft2(o, s=(H, W), dim=(1, 2), norm='ortho')
        return x.permute(0, 3, 1, 2).to(res.dtype) + res


class AFNOBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio, drop):
        super().__init__()
        self.n1 = nn.GroupNorm(min(32, dim), dim)
        self.afno = AFNO2D(dim, heads)
        self.n2 = nn.GroupNorm(min(32, dim), dim)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Conv2d(dim, h, 1), nn.GELU(),
                                 nn.Dropout2d(drop), nn.Conv2d(h, dim, 1))

    def forward(self, x):
        x = x + self.afno(self.n1(x))
        return x + self.mlp(self.n2(x))


class ConvBlock(nn.Module):
    """Two 3x3 convs, GroupNorm, GELU, residual. Identical in V4, ResGRU and FMG-WACLU."""
    def __init__(self, ci, co, drop=0.0):
        super().__init__()
        self.c1 = nn.Conv2d(ci, co, 3, padding=1, bias=False)
        self.g1 = nn.GroupNorm(min(32, co), co)
        self.c2 = nn.Conv2d(co, co, 3, padding=1, bias=False)
        self.g2 = nn.GroupNorm(min(32, co), co)
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
        self.skip = nn.Conv2d(ci, co, 1, bias=False) if ci != co else nn.Identity()

    def forward(self, x):
        r = self.skip(x)
        x = F.gelu(self.g1(self.c1(x)))
        x = self.drop(x)
        x = F.gelu(self.g2(self.c2(x)))
        return x + r


class Incep1D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        q = ch // 4
        self.bs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(ch, q, k, padding=k // 2), nn.GELU())
            for k in (1, 3, 5, 7)])
        self.fuse = nn.Conv1d(ch, ch, 1)
        self.norm = nn.GroupNorm(min(32, ch), ch)

    def forward(self, x):
        y = torch.cat([b(x) for b in self.bs], dim=1)
        return self.norm(self.fuse(y))


class Translator(nn.Module):
    def __init__(self, ch, t_in, t_out, n_layers):
        super().__init__()
        self.t_out = t_out
        self.enc = nn.ModuleList([Incep1D(ch) for _ in range(n_layers // 2)])
        self.up = nn.ConvTranspose1d(ch, ch, kernel_size=9, stride=1, padding=1)
        self.dec = nn.ModuleList([Incep1D(ch) for _ in range(n_layers // 2)])

    def forward(self, x):
        B, T, Ch, h, w = x.shape
        x = x.permute(0, 3, 4, 2, 1).reshape(B * h * w, Ch, T)
        for blk in self.enc:
            x = x + blk(x)
        x = self.up(x)
        for blk in self.dec:
            x = x + blk(x)
        return x.reshape(B, h, w, Ch, self.t_out).permute(0, 4, 3, 1, 2)


class AFNOv4(nn.Module):
    def __init__(self, cfg=C):
        super().__init__()
        e = cfg.ENC
        d = cfg.DROP
        self.cfg = cfg
        self.enc0 = ConvBlock(cfg.C_IN, e[0], d)
        self.enc1 = ConvBlock(e[0], e[1], d)
        self.enc2 = ConvBlock(e[1], e[2], d)
        self.pool = nn.MaxPool2d(2)
        self.pool_bn = nn.AvgPool2d(2, ceil_mode=True)
        self.mid_afno = nn.ModuleList([
            AFNOBlock(e[2], cfg.AFNO_HEADS, cfg.MLP_RATIO, d)
            for _ in range(cfg.MID_AFNO_BLOCKS)])
        self.afno = nn.ModuleList([
            AFNOBlock(e[2], cfg.AFNO_HEADS, cfg.MLP_RATIO, d)
            for _ in range(cfg.AFNO_BLOCKS)])
        self.trans = Translator(e[2], cfg.T_IN, cfg.T_OUT, cfg.TRANS_LAYERS)
        self.lead_emb = nn.Embedding(cfg.T_OUT, e[2])
        self.post_afno = nn.ModuleList([
            AFNOBlock(e[2], cfg.AFNO_HEADS, cfg.MLP_RATIO, d)
            for _ in range(cfg.POST_AFNO_BLOCKS)])
        self.fuse0 = nn.Conv2d(2 * e[0], e[0], 1, bias=False)
        self.fuse1 = nn.Conv2d(2 * e[1], e[1], 1, bias=False)
        self.fuse2 = nn.Conv2d(2 * e[2], e[2], 1, bias=False)
        dc = [128, 64, 32]
        self.up2 = nn.ConvTranspose2d(e[2], dc[0], 2, stride=2)
        self.dec2 = ConvBlock(dc[0] + e[2], dc[0], d)
        self.up1 = nn.ConvTranspose2d(dc[0], dc[1], 2, stride=2)
        self.dec1 = ConvBlock(dc[1] + e[1], dc[1], d)
        self.up0 = nn.ConvTranspose2d(dc[1], dc[2], 2, stride=2)
        self.dec0 = ConvBlock(dc[2] + e[0], dc[2], d)
        self.head = nn.Sequential(
            nn.Conv2d(dc[2], dc[2], 3, padding=1, bias=False),
            nn.GroupNorm(min(32, dc[2]), dc[2]), nn.GELU(),
            nn.Conv2d(dc[2], 1, 1))
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    @staticmethod
    def _pad(x, ref):
        dh, dw = ref.shape[2] - x.shape[2], ref.shape[3] - x.shape[3]
        if dh > 0 or dw > 0:
            return F.pad(x, [0, dw, 0, dh])
        if dh < 0 or dw < 0:
            return x[:, :, :ref.shape[2], :ref.shape[3]]
        return x

    def forward(self, x, last_log):
        B, Ch, T, H, W = x.shape
        xt = x.permute(0, 2, 1, 3, 4).reshape(B * T, Ch, H, W)
        e0 = self.enc0(xt)
        e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(self.pool(e1))
        for blk in self.mid_afno:
            e2 = blk(e2)
        bn = self.pool_bn(e2)
        for blk in self.afno:
            bn = blk(bn)

        def fuse(e, conv):
            e = e.reshape(B, T, *e.shape[1:])
            return conv(torch.cat([e[:, -1], e.mean(1)], dim=1))

        s0, s1, s2 = fuse(e0, self.fuse0), fuse(e1, self.fuse1), fuse(e2, self.fuse2)

        bn = bn.reshape(B, T, *bn.shape[1:])
        bn_out = self.trans(bn)
        bn_out = bn_out + self.lead_emb.weight[None, :, :, None, None]

        To = self.cfg.T_OUT
        bo = bn_out.reshape(B * To, *bn_out.shape[2:])
        for blk in self.post_afno:
            bo = blk(bo)
        rep = lambda s: s.unsqueeze(1).expand(-1, To, -1, -1, -1).reshape(
            B * To, *s.shape[1:])
        d2 = self.dec2(torch.cat([self._pad(self.up2(bo), rep(s2)), rep(s2)], 1))
        d1 = self.dec1(torch.cat([self._pad(self.up1(d2), rep(s1)), rep(s1)], 1))
        d0 = self.dec0(torch.cat([self._pad(self.up0(d1), rep(s0)), rep(s0)], 1))
        delta = self.head(d0).reshape(B, To, H, W)
        return last_log.unsqueeze(1) + delta


BACKBONES['afnov4'] = lambda: AFNOv4()

# %% [markdown]
# ## 4b. Network under test: {{TITLE}}
#
# {{ARCH}}
#
# The class below is the original code with only the changes listed at the top.

# %%
# BACKBONE-CELL: resgru fmgwaclu
class ConvGRUCell(nn.Module):
    """Convolutional GRU cell (notebook 30 and notebook 13 use the same math)."""
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        p = kernel_size // 2
        self.conv_gates = nn.Conv2d(input_dim + hidden_dim, 2 * hidden_dim, kernel_size, padding=p)
        self.conv_candidate = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=p)
        self.gn_gates = nn.GroupNorm(min(32, 2 * hidden_dim), 2 * hidden_dim)
        self.gn_cand = nn.GroupNorm(min(32, hidden_dim), hidden_dim)

    def forward(self, x, h):
        gates = torch.sigmoid(self.gn_gates(self.conv_gates(torch.cat([x, h], 1))))
        r, u = gates.chunk(2, dim=1)
        cand = torch.tanh(self.gn_cand(self.conv_candidate(torch.cat([x, r * h], 1))))
        return (1 - u) * h + u * cand

# %%
# BACKBONE-CELL: resgru
class ResGRUUNet(nn.Module):
    """Notebook 30 ResGRUUNet. Only change: input width is c_in x T_in channels."""
    def __init__(self, c_in, t_in=10, t_out=16, enc=(64, 128, 256), gru_hidden=256,
                 dec=(128, 64, 32), drop=0.1):
        super().__init__()
        self.t_out = t_out
        self.enc0 = ConvBlock(c_in * t_in, enc[0], drop)
        self.pool0 = nn.MaxPool2d(2)
        self.enc1 = ConvBlock(enc[0], enc[1], drop)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(enc[1], enc[2], drop)
        self.pool2 = nn.AvgPool2d(2, ceil_mode=True)
        self.bottleneck = ConvBlock(enc[2], gru_hidden, drop)
        self.convgru = ConvGRUCell(gru_hidden, gru_hidden)
        self.gru_proj = nn.Conv2d(gru_hidden, enc[2], 1, bias=False)
        self.up2 = nn.ConvTranspose2d(enc[2], dec[0], 2, stride=2)
        self.dec2 = ConvBlock(dec[0] + enc[2], dec[0], drop)
        self.up1 = nn.ConvTranspose2d(dec[0], dec[1], 2, stride=2)
        self.dec1 = ConvBlock(dec[1] + enc[1], dec[1], drop)
        self.up0 = nn.ConvTranspose2d(dec[1], dec[2], 2, stride=2)
        self.dec0 = ConvBlock(dec[2] + enc[0], dec[2], drop)
        self.head = nn.Sequential(
            nn.Conv2d(dec[2], dec[2], 3, padding=1, bias=False),
            nn.GroupNorm(min(32, dec[2]), dec[2]), nn.GELU(),
            nn.Conv2d(dec[2], 1, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    @staticmethod
    def _crop(x, ref):
        return x[:, :, :ref.shape[2], :ref.shape[3]]

    def forward(self, x, last_log):
        B, Ch, T, H, W = x.shape
        x = x.reshape(B, Ch * T, H, W)
        e0 = self.enc0(x)
        e1 = self.enc1(self.pool0(e0))
        e2 = self.enc2(self.pool1(e1))
        bn = self.bottleneck(self.pool2(e2))
        h = bn
        outputs = []
        for _ in range(self.t_out):
            h = self.convgru(bn, h)
            d = self.gru_proj(h)
            d = self._crop(self.up2(d), e2)
            d = self.dec2(torch.cat([d, e2], 1))
            d = self._crop(self.up1(d), e1)
            d = self.dec1(torch.cat([d, e1], 1))
            d = self._crop(self.up0(d), e0)
            d = self.dec0(torch.cat([d, e0], 1))
            outputs.append(self.head(d).squeeze(1))
        return last_log.unsqueeze(1) + torch.stack(outputs, dim=1)


BACKBONES['resgru'] = lambda: ResGRUUNet(C.C_IN)

# %%
# BACKBONE-CELL: fmgwaclu
class ConvLSTMCellGN(nn.Module):
    """Notebook 13 ConvLSTM cell (GroupNorm on the gates)."""
    def __init__(self, in_dim, hid_dim, ks=3):
        super().__init__()
        self.hid = hid_dim
        self.conv = nn.Conv2d(in_dim + hid_dim, 4 * hid_dim, ks, padding=ks // 2, bias=True)
        self.gn = nn.GroupNorm(min(32, 4 * hid_dim), 4 * hid_dim)

    def forward(self, x, state=None):
        B, _, H, W = x.shape
        h = x.new_zeros(B, self.hid, H, W) if state is None else state[0]
        c = x.new_zeros(B, self.hid, H, W) if state is None else state[1]
        gates = self.gn(self.conv(torch.cat([x, h], 1)))
        i, f, g, o = gates.chunk(4, 1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class WindPhysicsModule(nn.Module):
    """Adds wind speed, direction sin/cos and divergence channels: (B,C,H,W) -> (B,C+4,H,W)."""
    def __init__(self, u_idx, v_idx):
        super().__init__()
        self.u_idx = u_idx
        self.v_idx = v_idx

    def forward(self, x):
        u = x[:, self.u_idx:self.u_idx + 1]
        v = x[:, self.v_idx:self.v_idx + 1]
        mag = torch.sqrt(u ** 2 + v ** 2 + 1e-8)
        sd = u / (mag + 1e-8)
        cd = v / (mag + 1e-8)
        du = F.pad(u[:, :, :, 1:] - u[:, :, :, :-1], (0, 1))
        dv = F.pad(v[:, :, 1:, :] - v[:, :, :-1, :], (0, 0, 0, 1))
        return torch.cat([x, mag, sd, cd, du + dv], 1)


class MetForecaster(nn.Module):
    """ConvLSTM that rolls the input fields forward n_future steps from the 10 h history."""
    def __init__(self, in_ch, hid):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv2d(in_ch, hid, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, hid), hid), nn.GELU())
        self.cell = ConvLSTMCellGN(hid, hid)
        self.head = nn.Conv2d(hid, in_ch, 1)

    def forward(self, met_seq, n_future):
        state = None
        for t in range(met_seq.shape[1]):
            state = self.cell(self.embed(met_seq[:, t]), state)
        preds = []
        curr = met_seq[:, -1]
        for _ in range(n_future):
            state = self.cell(self.embed(curr), state)
            curr = self.head(state[0])
            preds.append(curr)
        return torch.stack(preds, 1)


class FMGWACLUNet(nn.Module):
    """Notebook 13 FMG-WACLU (future-met gating + wind-augmented ConvLSTM U-Net).
    Only change: input width c_in."""
    def __init__(self, c_in, t_out=16, enc=(32, 64, 128), hid=128, dec=(64, 32, 16),
                 drop=0.1, u_idx=3, v_idx=4):
        super().__init__()
        self.t_out = t_out
        self.wind = WindPhysicsModule(u_idx, v_idx)
        self.f_embed = nn.Sequential(
            nn.Conv2d(c_in + 4, enc[0], 3, padding=1, bias=False),
            nn.GroupNorm(min(32, enc[0]), enc[0]), nn.GELU())
        self.hist_lstm = ConvLSTMCellGN(enc[0], hid)
        self.met_fc = MetForecaster(c_in, enc[0])
        self.fut_embed = nn.Sequential(
            nn.Conv2d(c_in, enc[0], 3, padding=1, bias=False),
            nn.GroupNorm(min(32, enc[0]), enc[0]), nn.GELU())
        self.fut_lstm = ConvLSTMCellGN(enc[0], hid)
        self.gate_net = nn.Sequential(
            nn.Conv2d(hid * 2, hid, 1, bias=False),
            nn.GroupNorm(min(32, hid), hid), nn.Sigmoid())
        self.fuse_net = nn.Sequential(
            nn.Conv2d(hid * 2, hid, 1, bias=False),
            nn.GroupNorm(min(32, hid), hid), nn.GELU())
        self.pool0 = nn.MaxPool2d(2)
        self.enc1 = ConvBlock(hid, enc[1], drop)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(enc[1], enc[2], drop)
        self.pool2 = nn.AvgPool2d(2, ceil_mode=True)
        self.bottleneck = ConvBlock(enc[2], hid, drop)
        self.dec_gru = ConvGRUCell(hid, hid)
        self.gru_proj = nn.Conv2d(hid, enc[2], 1, bias=False)
        self.up2 = nn.ConvTranspose2d(enc[2], dec[0], 2, stride=2)
        self.dec2 = ConvBlock(dec[0] + enc[2], dec[0], drop)
        self.up1 = nn.ConvTranspose2d(dec[0], dec[1], 2, stride=2)
        self.dec1 = ConvBlock(dec[1] + enc[1], dec[1], drop)
        self.up0 = nn.ConvTranspose2d(dec[1], dec[2], 2, stride=2)
        self.dec0 = ConvBlock(dec[2] + hid, dec[2], drop)
        self.head = nn.Sequential(
            nn.Conv2d(dec[2], dec[2], 3, padding=1, bias=False),
            nn.GroupNorm(min(32, dec[2]), dec[2]), nn.GELU(),
            nn.Conv2d(dec[2], 1, 1))
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    @staticmethod
    def _pad(x, tgt):
        dh = tgt.shape[2] - x.shape[2]
        dw = tgt.shape[3] - x.shape[3]
        if dh > 0 or dw > 0:
            return F.pad(x, [0, dw, 0, dh])
        if dh < 0 or dw < 0:
            return x[:, :, :tgt.shape[2], :tgt.shape[3]]
        return x

    def forward(self, x, last_log):
        B, Ch, T, H, W = x.shape
        hs = None
        for t in range(T):
            hs = self.hist_lstm(self.f_embed(self.wind(x[:, :, t])), hs)
        h_hist = hs[0]
        fut_met = self.met_fc(x.permute(0, 2, 1, 3, 4), self.t_out)
        h_fut = self.fut_lstm(self.fut_embed(fut_met.mean(1)), None)[0]
        cat = torch.cat([h_hist, h_fut], 1)
        gate = self.gate_net(cat)
        h = gate * h_hist + (1 - gate) * self.fuse_net(cat)
        e0 = h
        e1 = self.enc1(self.pool0(e0))
        e2 = self.enc2(self.pool1(e1))
        bn = self.bottleneck(self.pool2(e2))
        gh = bn
        outputs = []
        for _ in range(self.t_out):
            gh = self.dec_gru(bn, gh)
            d = self.gru_proj(gh)
            d2 = self.dec2(torch.cat([self._pad(self.up2(d), e2), e2], 1))
            d1 = self.dec1(torch.cat([self._pad(self.up1(d2), e1), e1], 1))
            d0 = self.dec0(torch.cat([self._pad(self.up0(d1), e0), e0], 1))
            outputs.append(self.head(d0).squeeze(1))
        return last_log.unsqueeze(1) + torch.stack(outputs, 1)


BACKBONES['fmgwaclu'] = lambda: FMGWACLUNet(C.C_IN, u_idx=C.U_IDX, v_idx=C.V_IDX)

# %%
# BACKBONE-CELL: convlstm
CL_DROP, CL_SD = 0.25, 0.15   # dropout and stochastic depth of convlstm-v2


class AdvectionFeature(nn.Module):
    """Parameter-free semi-Lagrangian shift of the PM2.5 frame by the wind."""
    def __init__(self, scale=0.03):
        super().__init__()
        self.scale = float(scale)

    def forward(self, pm25, u10, v10):
        B, T, Hg, Wg = pm25.shape
        gy = torch.linspace(-1.0, 1.0, Hg, dtype=torch.float32, device=pm25.device)
        gx = torch.linspace(-1.0, 1.0, Wg, dtype=torch.float32, device=pm25.device)
        gy, gx = torch.meshgrid(gy, gx, indexing='ij')
        gy, gx = gy.unsqueeze(0), gx.unsqueeze(0)
        out = []
        for t in range(T):
            pm_t = pm25[:, t:t + 1].float()
            ny = torch.clamp(gy - v10[:, t].float() * self.scale, min=-1.0, max=1.0)
            nx = torch.clamp(gx - u10[:, t].float() * self.scale, min=-1.0, max=1.0)
            grid = torch.stack([nx, ny], dim=-1)
            out.append(F.grid_sample(pm_t, grid, mode='bilinear',
                                     padding_mode='border', align_corners=True))
        return torch.cat(out, dim=1)


class StochasticDepth(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.p = drop_prob

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        keep = 1 - self.p
        mask = (torch.rand((x.shape[0],) + (1,) * (x.ndim - 1), device=x.device) < keep).float()
        return x * mask / keep


class ConvBnGelu(nn.Module):
    def __init__(self, ic, oc, k=3, p=1, dr=CL_DROP):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ic, oc, k, padding=p, bias=False),
            nn.BatchNorm2d(oc), nn.GELU(), nn.Dropout2d(dr))

    def forward(self, x):
        return self.net(x)


class DoubleConv(nn.Module):
    def __init__(self, ic, oc, dp=CL_SD):
        super().__init__()
        self.net = nn.Sequential(ConvBnGelu(ic, oc), ConvBnGelu(oc, oc))
        self.bypass = nn.Conv2d(ic, oc, 1) if ic != oc else nn.Identity()
        self.sd = StochasticDepth(dp)

    def forward(self, x):
        return self.net(x) + self.sd(self.bypass(x))


class SpectralBlock2D(nn.Module):
    """FNO-style spectral convolution keeping the lowest m1 x m2 modes."""
    def __init__(self, ch, m1=8, m2=8):
        super().__init__()
        self.m1, self.m2 = m1, m2
        sc = 1.0 / ch ** 2
        self.wr = nn.Parameter(sc * torch.rand(ch, ch, m1, m2))
        self.wi = nn.Parameter(sc * torch.rand(ch, ch, m1, m2))
        self.bypass = nn.Conv2d(ch, ch, 1)
        self.drop = nn.Dropout2d(CL_DROP * 0.5)

    def forward(self, x):
        x32 = x.float()
        B, Ch, Hg, Wg = x32.shape
        xft = torch.fft.rfft2(x32, norm='ortho')
        out = torch.zeros_like(xft)
        m1, m2 = min(self.m1, xft.shape[-2]), min(self.m2, xft.shape[-1])
        w = torch.complex(self.wr[:, :, :m1, :m2], self.wi[:, :, :m1, :m2])
        out[:, :, :m1, :m2] = torch.einsum('bixy,ioxy->boxy', xft[:, :, :m1, :m2], w)
        xsp = torch.fft.irfft2(out, s=(Hg, Wg), norm='ortho').to(x.dtype)
        return F.gelu(self.drop(xsp) + self.bypass(x))


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, ks=3):
        super().__init__()
        self.hd = hidden_dim
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, ks, padding=ks // 2)

    def forward(self, x, h, c):
        i, f, o, g = self.conv(torch.cat([x, h], 1)).chunk(4, dim=1)
        c2 = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h2 = torch.sigmoid(o) * torch.tanh(c2)
        return h2, c2

    def init_hidden(self, B, H, W, dev, dtype):
        z = lambda: torch.zeros(B, self.hd, H, W, device=dev, dtype=dtype)
        return z(), z()


class ConvLSTMEncoder(nn.Module):
    def __init__(self, f_in, f_out):
        super().__init__()
        self.cell = ConvLSTMCell(f_in, f_out)
        self.bn = nn.BatchNorm2d(f_out)
        self.drop = nn.Dropout2d(CL_DROP)

    def forward(self, x):
        B, Fc, T, Hg, Wg = x.shape
        h, c = self.cell.init_hidden(B, Hg, Wg, x.device, x.dtype)
        for t in range(T):
            h, c = self.cell(x[:, :, t, :, :], h, c)
        return self.drop(self.bn(h))


class PM25ForecastNet(nn.Module):
    """convlstm-v2 PM25ForecastNet: ConvLSTM over the 10 input hours at full
    resolution, U-Net with BatchNorm, FNO spectral bottleneck. Changes: season
    FiLM removed, head output used as the change from the last frame."""
    def __init__(self, c_in, t_out=16, base_ch=32, u_idx=3, v_idx=4):
        super().__init__()
        ch = base_ch
        self.u_idx, self.v_idx = u_idx, v_idx
        self.adv = AdvectionFeature(0.03)
        self.temp_enc = ConvLSTMEncoder(c_in + 1, ch)
        self.enc1 = DoubleConv(ch, ch)
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(ch, ch * 2))
        self.enc3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(ch * 2, ch * 4))
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(ch * 4, ch * 8),
            SpectralBlock2D(ch * 8, 8, 8), SpectralBlock2D(ch * 8, 8, 8))
        self.up3 = nn.ConvTranspose2d(ch * 8, ch * 4, 2, stride=2)
        self.dec3 = DoubleConv(ch * 8, ch * 4)
        self.up2 = nn.ConvTranspose2d(ch * 4, ch * 2, 2, stride=2)
        self.dec2 = DoubleConv(ch * 4, ch * 2)
        self.up1 = nn.ConvTranspose2d(ch * 2, ch, 2, stride=2)
        self.dec1 = DoubleConv(ch * 2, ch)
        self.head = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.GELU(),
            nn.Dropout2d(CL_DROP * 0.5), nn.Conv2d(ch, t_out, 1))
        self._init_w()
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def _init_w(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @staticmethod
    def _crop(skip, tgt):
        if skip.shape[2:] == tgt.shape[2:]:
            return skip
        dh = skip.shape[2] - tgt.shape[2]
        dw = skip.shape[3] - tgt.shape[3]
        return skip[:, :, dh // 2:dh // 2 + tgt.shape[2], dw // 2:dw // 2 + tgt.shape[3]]

    def forward(self, x, last_log):
        adv = self.adv(x[:, 0], x[:, self.u_idx], x[:, self.v_idx]).unsqueeze(1)
        h = self.temp_enc(torch.cat([x, adv.to(x.dtype)], dim=1))
        e1 = self.enc1(h)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bottleneck(e3)
        u3 = self.up3(b)
        d3 = self.dec3(torch.cat([u3, self._crop(e3, u3)], 1))
        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, self._crop(e2, u2)], 1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, self._crop(e1, u1)], 1))
        delta = F.interpolate(self.head(d1), size=x.shape[-2:], mode='bilinear',
                              align_corners=False)
        return last_log.unsqueeze(1) + delta


BACKBONES['convlstm'] = lambda: PM25ForecastNet(C.C_IN, u_idx=C.U_IDX, v_idx=C.V_IDX)

# %%
# BACKBONE-CELL: patchafno
class PatchEmbed(nn.Module):
    """Fold time into channels, then a 4x4 patch projection."""
    def __init__(self, in_channels, embed_dim, patch_size, input_hours):
        super().__init__()
        self.proj = nn.Conv2d(in_channels * input_hours, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, Ch, T, H, W = x.shape
        x = self.proj(x.reshape(B, Ch * T, H, W))
        B, D, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x), Hp, Wp


class AdaptiveFourierBlock(nn.Module):
    """Notebook 35 spectral block: FFT -> one complex weight per head -> IFFT -> MLP."""
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        head_dim = dim // num_heads
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(dim)
        self.scale = nn.Parameter(torch.ones(1, 1, dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(1, 1, dim))
        self.freq_w_re = nn.Parameter(torch.randn(num_heads, head_dim, head_dim) * 0.02)
        self.freq_w_im = nn.Parameter(torch.randn(num_heads, head_dim, head_dim) * 0.02)
        self.sparsity = 0.01
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, dim), nn.Dropout(drop))

    def forward(self, x, Hp, Wp):
        # Notebook 35 called irfft2(xf.float()): .float() on a complex tensor
        # keeps only the real part (warning in its Kaggle log, cell line 60).
        # Here the complex spectrum goes to irfft2 intact, in float32.
        B, N, D = x.shape
        res = x
        x = self.norm1(x).reshape(B, Hp, Wp, D)
        with torch.autocast(x.device.type, enabled=False):
            xf = torch.fft.rfft2(x.float(), dim=(1, 2), norm='ortho')
            bx, hx, wx, dx = xf.shape
            hd = dx // self.num_heads
            xf = xf.reshape(bx, hx, wx, self.num_heads, hd)
            o_re = (torch.einsum('bhwnd,nde->bhwne', xf.real, self.freq_w_re)
                    - torch.einsum('bhwnd,nde->bhwne', xf.imag, self.freq_w_im))
            o_im = (torch.einsum('bhwnd,nde->bhwne', xf.real, self.freq_w_im)
                    + torch.einsum('bhwnd,nde->bhwne', xf.imag, self.freq_w_re))
            xf = torch.complex(F.softshrink(o_re, self.sparsity),
                               F.softshrink(o_im, self.sparsity)).reshape(bx, hx, wx, dx)
            x = torch.fft.irfft2(xf, s=(Hp, Wp), dim=(1, 2), norm='ortho')
        x = x.reshape(B, N, D).to(res.dtype) * self.scale + self.bias + res
        return x + self.mlp(self.norm2(x))


class TemporalProjectionHead(nn.Module):
    """Linear projection of each token to T_out x 4 x 4 values, then unpatchify."""
    def __init__(self, embed_dim, output_hours, patch_size, img_h, img_w):
        super().__init__()
        self.out_t = output_hours
        self.ph, self.pw = patch_size
        self.Hp = img_h // patch_size[0]
        self.Wp = img_w // patch_size[1]
        self.proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, output_hours * patch_size[0] * patch_size[1]))

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x).reshape(B, self.Hp, self.Wp, self.out_t, self.ph, self.pw)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        return x.reshape(B, self.out_t, self.Hp * self.ph, self.Wp * self.pw)


class PatchAFNO(nn.Module):
    """Notebook 35 AFNO_PM25. Changes: input width c_in, output used as the
    change from the last frame (last projection zero-initialized)."""
    def __init__(self, c_in, t_in=10, t_out=16, h=140, w=124, dim=256, depth=8,
                 heads=4, mlp_ratio=4.0, drop=0.1, patch=(4, 4)):
        super().__init__()
        self.patch_embed = PatchEmbed(c_in, dim, patch, t_in)
        self.pos_embed = nn.Parameter(torch.zeros(1, (h // patch[0]) * (w // patch[1]), dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([
            AdaptiveFourierBlock(dim, heads, mlp_ratio, drop) for _ in range(depth)])
        self.head = TemporalProjectionHead(dim, t_out, patch, h, w)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.head.proj[-1].weight)
        nn.init.zeros_(self.head.proj[-1].bias)

    def forward(self, x, last_log):
        x, Hp, Wp = self.patch_embed(x)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x, Hp, Wp)
        return last_log.unsqueeze(1) + self.head(x)


BACKBONES['patchafno'] = lambda: PatchAFNO(C.C_IN)

