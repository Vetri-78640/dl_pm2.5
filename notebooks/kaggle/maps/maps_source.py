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
# | seeds | every model of the study, loaded from earlier runs (no training here) |
# | post-processing | 2- and 5-seed means; V4's per-horizon calibration (section 11) |
#
# **Round-one reference (not retrained):** AFNO V4 seed 42 phase A best, val 0.9110,
# private 0.8762 without calibration. Persistence: val 0.8515.
#
# **Round-one decision rule:** AFNO V4 is called better than
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

# %% [markdown]
# ## 5. Loss and the exact competition metric (V4, unchanged)
#
# Training loss: Huber in log1p space + global SMAPE + episode SMAPE +
# (1 - episode Pearson), with horizon weights 0.7 -> 1.3. The validation
# metric is the competition score with equal weights; `evaluate` also reports
# each term per forecast hour.

# %%
EPS = 1e-8

def horizon_weights(device):
    w = torch.linspace(C.HW_MIN, C.HW_MAX, C.T_OUT, device=device)
    return w / w.mean()


def _masked_corr(pred, tgt, mask, min_pts=8):
    B, T = pred.shape[:2]
    p = pred.reshape(B, T, -1)
    y = tgt.reshape(B, T, -1)
    m = mask.reshape(B, T, -1)
    n = m.sum(-1)
    valid = n >= min_pts
    n = n.clamp(min=1)
    mp = (p * m).sum(-1) / n
    my = (y * m).sum(-1) / n
    pc = (p - mp[..., None]) * m
    yc = (y - my[..., None]) * m
    cov = (pc * yc).sum(-1)
    vp = (pc ** 2).sum(-1)
    vy = (yc ** 2).sum(-1)
    corr = cov / torch.sqrt(vp.clamp(min=1e-4) * vy.clamp(min=1e-4))
    return corr, valid


class CompositeLoss(nn.Module):
    def __init__(self, w, ep_wt):
        super().__init__()
        self.w = w
        self.ep_wt = ep_wt
        self.hw = None

    def forward(self, pred_log, t_log, t_raw, mask):
        if self.hw is None or self.hw.device != pred_log.device:
            self.hw = horizon_weights(pred_log.device)
        hw = self.hw[None, :, None, None]
        pred_raw = F.relu(torch.expm1(pred_log.clamp(max=13.0)))
        wpx = (1.0 + (self.ep_wt - 1.0) * mask) * hw
        hub = F.smooth_l1_loss(pred_log, t_log, beta=1.0, reduction='none')
        L_h = (hub * wpx).mean()
        sm = (pred_raw - t_raw).abs() / (0.5 * (pred_raw.abs() + t_raw.abs()) + 0.1)
        L_g = (sm * hw).mean()
        L_es = (sm * mask).sum() / (mask.sum() + EPS)
        corr, valid = _masked_corr(pred_raw, t_raw, mask)
        L_ec = 1.0 - (corr * valid).sum() / (valid.sum() + EPS)
        loss = (self.w['huber'] * L_h + self.w['gsmape'] * L_g
                + self.w['esmape'] * L_es + self.w['ecorr'] * L_ec)
        with torch.no_grad():
            mets = dict(gsmape=sm.mean().item(), esmape=L_es.item(),
                        ecorr=1.0 - L_ec.item())
        return loss, mets


@torch.no_grad()
def comp_metric_batch(pred_raw, t_raw, mask, acc):
    sm = (pred_raw - t_raw).abs() / (0.5 * (pred_raw.abs() + t_raw.abs()) + EPS)
    acc['gsmape'].append(sm.mean(dim=(2, 3)).reshape(-1).cpu())
    m = mask
    n = m.sum(dim=(2, 3))
    es = (sm * m).sum(dim=(2, 3)) / n.clamp(min=1)
    has = n > 0
    acc['esmape'].append(es[has].reshape(-1).cpu())
    corr, valid = _masked_corr(pred_raw, t_raw, m, min_pts=3)
    acc['ecorr'].append(corr[valid].reshape(-1).cpu())


def comp_metric_final(acc):
    g = torch.cat(acc['gsmape']).mean().item()
    e = torch.cat(acc['esmape']).mean().item() if acc['esmape'] else 2.0
    c = torch.cat(acc['ecorr']).mean().item() if acc['ecorr'] else 0.0
    ng, nc, ne = 1 - g / 2, (c + 1) / 2, 1 - e / 2
    return dict(gsmape=g, esmape=e, ecorr=c,
                norm_gsmape=ng, norm_ecorr=nc, norm_esmape=ne,
                score=(ng + nc + ne) / 3)


@torch.no_grad()
def evaluate(model, loader):
    """Same score as validate(), plus each metric term per forecast hour."""
    model.eval()
    acc = defaultdict(list)
    g_t, e_t, c_t = [], [], []
    for x, t_log, t_raw, last_log, mask in loader:
        x, last_log = x.to(DEVICE), last_log.to(DEVICE)
        t_raw, mask = t_raw.to(DEVICE), mask.to(DEVICE)
        with torch.autocast('cuda', enabled=DEVICE.type == 'cuda'):
            pred_log = model(x, last_log)
        pred_raw = F.relu(torch.expm1(pred_log.float().clamp(max=13.0)))
        comp_metric_batch(pred_raw, t_raw, mask, acc)
        sm = (pred_raw - t_raw).abs() / (0.5 * (pred_raw.abs() + t_raw.abs()) + EPS)
        n = mask.sum(dim=(2, 3))
        e = (sm * mask).sum(dim=(2, 3)) / n.clamp(min=1)
        corr, valid = _masked_corr(pred_raw, t_raw, mask, min_pts=3)
        nan = lambda a: torch.full_like(a, float('nan'))
        g_t.append(sm.mean(dim=(2, 3)).cpu())
        e_t.append(torch.where(n > 0, e, nan(e)).cpu())
        c_t.append(torch.where(valid, corr, nan(corr)).cpu())
    mets = comp_metric_final(acc)
    per = lambda a: [round(v, 5) for v in torch.nanmean(torch.cat(a), dim=0).tolist()]
    mets['per_hour'] = dict(gsmape=per(g_t), esmape=per(e_t), ecorr=per(c_t))
    return mets


# %% [markdown]
# ## 6. Training - V4 phase A loop with gradient accumulation
#
# The loop is V4's (AdamW, one-cycle, AMP, grad clip 1.0, EMA ramp, best
# epoch by val score, last checkpoint every epoch, resume). The one addition:
# if a batch of 4 does not fit in GPU memory, the batch is split into
# micro-batches and gradients are accumulated, so the optimizer still sees an
# effective batch of 4 and the same number of steps per epoch.

# %%
class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.n = 0
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.float(), alpha=1 - d)

    def state_for_eval(self, model):
        sd = {k: v.clone() for k, v in model.state_dict().items()}
        for k in self.shadow:
            sd[k] = self.shadow[k].to(sd[k].dtype)
        return sd


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    acc = defaultdict(list)
    vloss, nb = 0.0, 0
    for x, t_log, t_raw, last_log, mask in loader:
        x, t_log = x.to(DEVICE), t_log.to(DEVICE)
        t_raw, last_log, mask = t_raw.to(DEVICE), last_log.to(DEVICE), mask.to(DEVICE)
        with torch.autocast('cuda', enabled=DEVICE.type == 'cuda'):
            pred_log = model(x, last_log)
        pred_log = pred_log.float()
        loss, _ = criterion(pred_log, t_log, t_raw, mask)
        vloss += loss.item(); nb += 1
        pred_raw = F.relu(torch.expm1(pred_log.clamp(max=13.0)))
        comp_metric_batch(pred_raw, t_raw, mask, acc)
    mets = comp_metric_final(acc)
    mets['loss'] = vloss / max(nb, 1)
    return mets


def time_left():
    return C.MAX_HOURS * 3600 - (time.time() - T0_GLOBAL)


def log_epoch(row):
    import csv
    path = os.path.join(C.WORK, 'train_log.csv')
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def pick_micro(model, ds, batch):
    """Largest micro-batch (batch, batch/2, ..., 1) whose forward + backward
    peaks below 85% of GPU memory. Model weights are restored afterwards."""
    if DEVICE.type != 'cuda':
        return batch
    state = {k: v.clone() for k, v in model.state_dict().items()}
    total = torch.cuda.get_device_properties(0).total_memory
    crit = CompositeLoss(C.LOSS_A, C.EP_WT_A)
    micro, chosen = batch, None
    while micro >= 1 and chosen is None:
        try:
            items = [ds[i] for i in range(micro)]
            x, t_log, t_raw, last_log, mask = [torch.stack(z).to(DEVICE) for z in zip(*items)]
            torch.cuda.reset_peak_memory_stats()
            model.train()
            with torch.autocast('cuda'):
                pl = model(x, last_log)
            loss, _ = crit(pl.float(), t_log, t_raw, mask)
            loss.backward()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            ok = peak < 0.85 * total
            print(f'  micro-batch {micro}: peak {peak/1e9:.2f} GB of {total/1e9:.1f} GB'
                  f' -> {"ok" if ok else "too close to the limit"}')
            if ok:
                chosen = micro
        except torch.cuda.OutOfMemoryError:
            print(f'  micro-batch {micro}: out of memory')
        x = t_log = t_raw = last_log = mask = pl = loss = None
        model.zero_grad(set_to_none=True)
        gc.collect(); torch.cuda.empty_cache()
        if chosen is None:
            micro //= 2
    model.load_state_dict(state)
    if chosen is None:
        raise RuntimeError('even a micro-batch of 1 does not fit in GPU memory')
    return chosen


def run_phase(model, name, train_ds, val_ds, loss_w, ep_wt, lr, epochs,
              batch, patience, micro):
    accum = batch // micro
    crit = CompositeLoss(loss_w, ep_wt)
    nw = C.WORKERS
    pw = nw > 0
    tl = DataLoader(train_ds, batch_size=micro, shuffle=True,
                    num_workers=nw, pin_memory=True, drop_last=True,
                    persistent_workers=pw)
    vl = DataLoader(val_ds, batch_size=micro, shuffle=False,
                    num_workers=nw, pin_memory=True, persistent_workers=pw)
    steps_ep = len(tl) // accum

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=C.WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=steps_ep * epochs, pct_start=0.05,
        div_factor=20, final_div_factor=200)
    scaler = torch.amp.GradScaler('cuda', enabled=DEVICE.type == 'cuda')
    ema = EMA(model, C.EMA_DECAY)
    best_path = os.path.join(C.WORK, f'{name}_best.pt')
    last_path = os.path.join(C.WORK, f'{name}_last.pt')

    start_ep, best_score, best_ep, bad = 1, -1.0, 0, 0
    time_stop = False
    prev_last = art(f'{name}_last.pt')
    if prev_last:
        ck = torch.load(prev_last, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched'])
        ema.shadow = {k: v.to(DEVICE) for k, v in ck['ema'].items()}
        ema.n = ck.get('ema_n', 0)
        start_ep = ck['epoch'] + 1
        best_score = ck['best_score']
        best_ep = ck.get('best_ep', 0)
        bad = ck.get('bad', 0)
        print(f'[{name}] resumed at epoch {start_ep}, best={best_score:.4f}')

    print(f'[{name}] {steps_ep} optimizer steps per epoch, micro-batch {micro} x {accum}')
    epoch_secs = []
    for ep in range(start_ep, epochs + 1):
        if time_left() < 600:
            print(f'[{name}] time budget reached, stopping')
            time_stop = True
            break
        t0 = time.time()
        if DEVICE.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        model.train()
        tloss, nb = 0.0, 0
        tr_acc = defaultdict(float)
        opt.zero_grad(set_to_none=True)
        grads_in_group = 0
        for i, (x, t_log, t_raw, last_log, mask) in enumerate(tl):
            if i >= steps_ep * accum:
                break
            if i % 50 == 0 and i > 0 and time_left() < 600:
                print(f'[{name}] time budget reached mid-epoch, stopping')
                time_stop = True
                break
            x, t_log = x.to(DEVICE, non_blocking=True), t_log.to(DEVICE, non_blocking=True)
            t_raw = t_raw.to(DEVICE, non_blocking=True)
            last_log = last_log.to(DEVICE, non_blocking=True)
            mask = mask.to(DEVICE, non_blocking=True)
            with torch.autocast('cuda', enabled=DEVICE.type == 'cuda'):
                pred_log = model(x, last_log)
            loss, mets = crit(pred_log.float(), t_log, t_raw, mask)
            if torch.isfinite(loss):
                scaler.scale(loss / accum).backward()
                grads_in_group += 1
                tloss += loss.item(); nb += 1
                for k, v in mets.items():
                    tr_acc[k] += v
            else:
                print(f'  [{name} E{ep} {i}] non-finite loss, micro-batch skipped')
            if (i + 1) % accum:
                continue
            step = (i + 1) // accum
            if grads_in_group:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                ema.update(model)
            opt.zero_grad(set_to_none=True)
            grads_in_group = 0
            if (step - 1) % 100 == 0 and nb:
                print(f'  [{name} E{ep} {step - 1}/{steps_ep}] loss={loss.item():.4f} '
                      f'gsm={mets["gsmape"]:.4f} esm={mets["esmape"]:.4f} '
                      f'ec={mets["ecorr"]:.4f} '
                      f'lr={opt.param_groups[0]["lr"]:.2e}')

        if time_stop:
            break
        raw_sd = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema.state_for_eval(model))
        vm = validate(model, vl, crit)
        model.load_state_dict(raw_sd)

        improved = vm['score'] > best_score
        if improved:
            best_score, best_ep, bad = vm['score'], ep, 0
            torch.save({'model': ema.state_for_eval(model), 'metrics': vm,
                        'epoch': ep, 'backbone': BACKBONE}, best_path)
        else:
            bad += 1
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(),
                    'sched': sched.state_dict(), 'ema': ema.shadow,
                    'ema_n': ema.n, 'epoch': ep, 'best_score': best_score,
                    'best_ep': best_ep, 'bad': bad}, last_path)
        tr_m = {k: v / max(nb, 1) for k, v in tr_acc.items()}
        secs = time.time() - t0
        epoch_secs.append(secs)
        peak = torch.cuda.max_memory_allocated() / 1e9 if DEVICE.type == 'cuda' else 0.0
        eta = (epochs - ep) * float(np.mean(epoch_secs)) / 60
        print(f'[{name}] E{ep}/{epochs} ({secs:.0f}s, peak {peak:.1f} GB, '
              f'~{eta:.0f} min left)')
        print(f'  train | loss={tloss/max(nb,1):.4f} '
              f'gsmape={tr_m.get("gsmape",0):.4f} '
              f'esmape={tr_m.get("esmape",0):.4f} '
              f'ecorr={tr_m.get("ecorr",0):.4f}')
        print(f'  val   | score={vm["score"]:.4f}  '
              f'NormGlobalSMAPE={vm["norm_gsmape"]:.4f} '
              f'NormEpisodeCorr={vm["norm_ecorr"]:.4f} '
              f'NormEpisodeSMAPE={vm["norm_esmape"]:.4f}  '
              f'(raw: gsm={vm["gsmape"]:.4f} esm={vm["esmape"]:.4f} '
              f'ec={vm["ecorr"]:.4f})'
              + ('  *BEST -> saved*' if improved else f'  (best={best_score:.4f})'))
        log_epoch(dict(phase=name, epoch=ep, secs=round(secs), peak_gb=round(peak, 2),
                       lr=opt.param_groups[0]['lr'],
                       tr_loss=round(tloss / max(nb, 1), 5),
                       tr_gsmape=round(tr_m.get('gsmape', 0), 5),
                       tr_esmape=round(tr_m.get('esmape', 0), 5),
                       tr_ecorr=round(tr_m.get('ecorr', 0), 5),
                       val_loss=round(vm['loss'], 5),
                       val_gsmape=round(vm['gsmape'], 5),
                       val_esmape=round(vm['esmape'], 5),
                       val_ecorr=round(vm['ecorr'], 5),
                       norm_gsmape=round(vm['norm_gsmape'], 5),
                       norm_ecorr=round(vm['norm_ecorr'], 5),
                       norm_esmape=round(vm['norm_esmape'], 5),
                       val_score=round(vm['score'], 5),
                       best=int(improved)))
        if bad >= patience:
            print(f'[{name}] early stop (best={best_score:.4f})')
            break

    if not time_stop:
        with open(os.path.join(C.WORK, f'{name}_done.flag'), 'w') as fh:
            fh.write(f'best={best_score:.4f} epoch={best_ep}\n')
    secs_mean = float(np.mean(epoch_secs)) if epoch_secs else None
    return (best_path if os.path.exists(best_path) else art(f'{name}_best.pt')), secs_mean


# %% [markdown]
# ## 7. Run: GPU check -> data -> pipeline check -> train -> evaluate -> predict
#
# Outputs in `/kaggle/working`: `preds.npy` (submission, no post-processing),
# `<backbone>_s42_phaseA_best.pt` / `_last.pt`, `train_log.csv` and
# `summary.json` (scores, per-hour metrics, pipeline check, timing).

# %%
def predict(model, test_ds, bs):
    model.eval()
    dl = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=True)
    out = np.zeros((len(test_ds), C.H, C.W, C.T_OUT), np.float32)
    i = 0
    with torch.no_grad():
        for x, last_log in dl:
            x, last_log = x.to(DEVICE), last_log.to(DEVICE)
            with torch.autocast('cuda', enabled=DEVICE.type == 'cuda'):
                pl = model(x, last_log)
            pr = F.relu(torch.expm1(pl.float().clamp(max=13.0)))
            out[i:i + x.shape[0]] = pr.permute(0, 2, 3, 1).cpu().numpy()
            i += x.shape[0]
    return out


def save_preds(preds):
    preds = preds.astype(np.float32)
    assert preds.shape == (C.N_TEST, C.H, C.W, C.T_OUT), preds.shape
    assert np.isfinite(preds).all()
    np.save(os.path.join(C.WORK, 'preds.npy'), preds)
    print(f'  preds.npy written  range=[{preds.min():.2f},{preds.max():.2f}]  '
          f'mean={preds.mean():.3f}')


def pipeline_check(val_ds):
    """Re-evaluate V4's seed-42 checkpoint here; it must reproduce its logged score."""
    path = art('s42_phaseA_best.pt')
    if not path:
        raise RuntimeError('V4 checkpoint s42_phaseA_best.pt not found in the attached V4 output')
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    ref = float(ck['metrics']['score'])
    model = AFNOv4().to(DEVICE)
    model.load_state_dict(ck['model'])
    vl = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=C.WORKERS, pin_memory=True)
    vm = evaluate(model, vl)
    diff = vm['score'] - ref
    print(f'  V4 seed-42 checkpoint: logged val {ref:.4f}, re-evaluated here {vm["score"]:.4f} '
          f'(diff {diff:+.5f}, tolerance {C.CHECK_TOL})')
    del model
    gc.collect(); torch.cuda.empty_cache()
    if abs(diff) > C.CHECK_TOL:
        raise RuntimeError('pipeline check failed: this notebook does not reproduce the V4 '
                           'validation score, so its results would not be comparable')
    return dict(logged=round(ref, 5), reproduced=round(vm['score'], 5),
                diff=round(diff, 6), v4_per_hour=vm['per_hour'],
                v4_terms={k: round(vm[k], 5) for k in ('gsmape', 'esmape', 'ecorr')})


def main():
    summary = dict(backbone=BACKBONE, seed=C.SEED, status='started',
                   torch=torch.__version__, v4_reference=C.V4_REF)
    sp = os.path.join(C.WORK, 'summary.json')
    dump = lambda: json.dump(summary, open(sp, 'w'), indent=1)

    print('=' * 60, '\nSTEP 0: environment\n', '=' * 60)
    if C.IS_KAGGLE:
        summary['gpu'] = check_gpu()
        if not C.DATA_ROOT:
            raise RuntimeError('competition data not found under /kaggle/input')
        if not C.V4_DIR:
            raise RuntimeError('V4 output not found: attach notebook imvetri0/afno-v4 as input')
    print('  data   :', C.DATA_ROOT)
    print('  V4 out :', C.V4_DIR)
    print('  resume :', C.RESUME_DIR)
    name = f'{BACKBONE}_s{C.SEED}_phaseA'
    if C.RESUME_DIR:
        for f in (f'{name}_best.pt', 'train_log.csv'):
            src = os.path.join(C.RESUME_DIR, f)
            if os.path.exists(src) and not os.path.exists(os.path.join(C.WORK, f)):
                shutil.copy(src, C.WORK)
                print(f'  carried over from the earlier run: {f}')

    print('=' * 60, '\nSTEP 1: stats + episode masks (from the V4 output)\n', '=' * 60)
    print('  stats from:', art('stats.npz') or 'computing')
    stats = build_stats()
    months, static = load_months(stats)

    print('=' * 60, '\nSTEP 2: datasets\n', '=' * 60)
    train_ds = PM25Windows(months, static, stats, 'train')
    val_ds = PM25Windows(months, static, stats, 'val')
    test_ds = PM25Test(stats, static)

    print('=' * 60, '\nSTEP 3: pipeline check (V4 checkpoint on this validation set)\n', '=' * 60)
    summary['pipeline_check'] = pipeline_check(val_ds)
    dump()

    print('=' * 60, f'\nSTEP 4: build {BACKBONE}\n', '=' * 60)
    seed_all(C.SEED)
    model = BACKBONES[BACKBONE]().to(DEVICE)
    n_par = sum(p.numel() for p in model.parameters())
    print(f'  {BACKBONE} params: {n_par:,} ({n_par/1e6:.2f}M)')
    x, _, _, last_log, _ = val_ds[0]
    model.eval()
    with torch.no_grad():
        y = model(x[None].to(DEVICE), last_log[None].to(DEVICE))
    assert y.shape == (1, C.T_OUT, C.H, C.W), y.shape
    gap = (y - last_log[None, None].to(DEVICE)).abs().max().item()
    print(f'  output {tuple(y.shape)}; at init it equals persistence (max diff {gap:.2e})')
    assert gap < 1e-4, 'final layer is not zero-initialized'
    micro = pick_micro(model, train_ds, C.A_BATCH)
    summary.update(params=n_par, micro_batch=micro, accum=C.A_BATCH // micro)
    dump()

    print('=' * 60, f'\nSTEP 5: train {name}\n', '=' * 60)
    secs = None
    t_train = time.time()
    if art(f'{name}_done.flag') and art(f'{name}_best.pt'):
        print('  already trained in an earlier session, skipping')
    else:
        try:
            _, secs = run_phase(model, name, train_ds, val_ds, C.LOSS_A, C.EP_WT_A,
                                C.A_LR, C.A_EPOCHS, C.A_BATCH, C.A_PATIENCE, micro)
        except Exception as e:
            traceback.print_exc()
            summary['status'] = f'training interrupted: {type(e).__name__}: {e}'
            dump()
    summary.update(train_minutes=round((time.time() - t_train) / 60, 1),
                   secs_per_epoch=round(secs, 1) if secs else None)
    if summary['status'] == 'started' and not art(f'{name}_done.flag'):
        summary['status'] = ('incomplete: time budget reached; attach this output '
                             'as an input and run again to resume')

    print('=' * 60, '\nSTEP 6: evaluate the best epoch + test inference\n', '=' * 60)
    best = art(f'{name}_best.pt')
    if not best:
        summary['status'] = 'failed: no checkpoint'
        dump()
        raise RuntimeError('no best checkpoint was saved')
    ck = torch.load(best, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck['model'])
    vl = DataLoader(val_ds, batch_size=micro, shuffle=False, num_workers=C.WORKERS, pin_memory=True)
    vm = evaluate(model, vl)
    print(f'  best epoch {ck.get("epoch")}: val score {vm["score"]:.4f} '
          f'(gsm {vm["gsmape"]:.4f}, esm {vm["esmape"]:.4f}, ec {vm["ecorr"]:.4f})')
    print(f'  AFNO V4 reference: val {C.V4_REF["val"]:.4f} -> difference '
          f'{vm["score"] - C.V4_REF["val"]:+.4f}')
    summary['best'] = dict(epoch=ck.get('epoch'),
                           **{k: round(v, 5) for k, v in vm.items() if k != 'per_hour'},
                           per_hour=vm['per_hour'])
    save_preds(predict(model, test_ds, micro))
    if summary['status'] == 'started':
        summary['status'] = 'complete'
    summary['total_minutes'] = round((time.time() - T0_GLOBAL) / 60, 1)
    dump()
    print('  summary.json written; status:', summary['status'])

# %% [markdown]
# ## 9. Ablations: why does ResGRU-UNet beat AFNO V4?
#
# Each variant changes one thing in one finalist and is trained once (seed 42) with
# the same phase A recipe. Hypotheses, predictions and the decision rule were fixed
# before these runs.
#
# | variant | hypothesis | change | prediction if the hypothesis holds |
# |---|---|---|---|
# | `afnov4_allskip` | H1: full-resolution time information | AFNO V4 skips fuse all 10 hours (1x1 conv) instead of [last hour, mean] | AFNO V4 improves, mainly global SMAPE |
# | `afnov4_pad` | H2: spectral wrap-around | every AFNO block runs on a zero-padded 2H x 2W map, cropped back | AFNO V4 improves |
# | `afnov4_nosparse` | H3: soft-thresholding removes detail | AFNO soft-threshold 0.01 -> 0 | AFNO V4 improves, mainly global SMAPE |
# | `resgru_norec` | H4: lead-time recurrence | ResGRU-UNet's ConvGRU replaced by a per-lead embedding + 3 ConvBlocks (no recurrence, same parameter count within 0.1%) | ResGRU-UNet gets worse, mainly episode correlation |

# %%
class AFNO2DPad(AFNO2D):
    """AFNO2D on a map zero-padded to 2H x 2W, cropped back: opposite edges no longer wrap."""
    def forward(self, x):
        H, W = x.shape[-2:]
        return super().forward(F.pad(x, [0, W, 0, H]))[..., :H, :W]


def afno_variant(pad=False, sparsity=None):
    m = AFNOv4()
    for blk in m.modules():
        if isinstance(blk, AFNOBlock):
            if pad:
                old = blk.afno
                new = AFNO2DPad(old.nb * old.bs, old.nb, old.sparsity)
                new.load_state_dict(old.state_dict())
                blk.afno = new
            if sparsity is not None:
                blk.afno.sparsity = sparsity
    return m


def widen_fuse(old, T):
    """1x1 conv on all T hours that equals `old` on [last hour, mean] at init."""
    w = old.weight.data
    e = w.shape[0]
    nw = (w[:, e:] / T).repeat(1, T, 1, 1)
    nw[:, (T - 1) * e:] += w[:, :e]
    new = nn.Conv2d(T * e, e, 1, bias=False)
    new.weight.data.copy_(nw)
    return new


class AFNOv4AllSkip(AFNOv4):
    """AFNO V4 whose skip connections keep every input hour (H1).
    At init it computes exactly what AFNO V4 computes; training can then weight each hour."""
    def __init__(self, cfg=C):
        super().__init__(cfg)
        self.fuse0 = widen_fuse(self.fuse0, cfg.T_IN)
        self.fuse1 = widen_fuse(self.fuse1, cfg.T_IN)
        self.fuse2 = widen_fuse(self.fuse2, cfg.T_IN)

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
            return conv(e.reshape(B, T * e.shape[1], *e.shape[2:]))

        s0, s1, s2 = fuse(e0, self.fuse0), fuse(e1, self.fuse1), fuse(e2, self.fuse2)
        bn = bn.reshape(B, T, *bn.shape[1:])
        bn_out = self.trans(bn) + self.lead_emb.weight[None, :, :, None, None]
        To = self.cfg.T_OUT
        bo = bn_out.reshape(B * To, *bn_out.shape[2:])
        for blk in self.post_afno:
            bo = blk(bo)
        rep = lambda s: s.unsqueeze(1).expand(-1, To, -1, -1, -1).reshape(B * To, *s.shape[1:])
        d2 = self.dec2(torch.cat([self._pad(self.up2(bo), rep(s2)), rep(s2)], 1))
        d1 = self.dec1(torch.cat([self._pad(self.up1(d2), rep(s1)), rep(s1)], 1))
        d0 = self.dec0(torch.cat([self._pad(self.up0(d1), rep(s0)), rep(s0)], 1))
        return last_log.unsqueeze(1) + self.head(d0).reshape(B, To, H, W)


class ResGRUNoRec(ResGRUUNet):
    """ResGRU-UNet without recurrence: each lead hour = 3 ConvBlocks(bottleneck + learned lead embedding) (H4).
    3 blocks keep the parameter count within 0.1% of ResGRU-UNet, so capacity is not the change."""
    def __init__(self, c_in, gru_hidden=256):
        super().__init__(c_in, gru_hidden=gru_hidden)
        del self.convgru
        self.lead_emb = nn.Parameter(torch.zeros(self.t_out, gru_hidden, 1, 1))
        self.lead_block = nn.Sequential(*[ConvBlock(gru_hidden, gru_hidden, 0.1) for _ in range(3)])

    def forward(self, x, last_log):
        B, Ch, T, H, W = x.shape
        x = x.reshape(B, Ch * T, H, W)
        e0 = self.enc0(x)
        e1 = self.enc1(self.pool0(e0))
        e2 = self.enc2(self.pool1(e1))
        bn = self.bottleneck(self.pool2(e2))
        outputs = []
        for k in range(self.t_out):
            d = self.gru_proj(self.lead_block(bn + self.lead_emb[k]))
            d = self._crop(self.up2(d), e2)
            d = self.dec2(torch.cat([d, e2], 1))
            d = self._crop(self.up1(d), e1)
            d = self.dec1(torch.cat([d, e1], 1))
            d = self._crop(self.up0(d), e0)
            d = self.dec0(torch.cat([d, e0], 1))
            outputs.append(self.head(d).squeeze(1))
        return last_log.unsqueeze(1) + torch.stack(outputs, dim=1)


BACKBONES['afnov4_allskip'] = lambda: AFNOv4AllSkip()
BACKBONES['afnov4_pad'] = lambda: afno_variant(pad=True)
BACKBONES['afnov4_nosparse'] = lambda: afno_variant(sparsity=0.0)
BACKBONES['resgru_norec'] = lambda: ResGRUNoRec(C.C_IN)

# %% [markdown]
# ## 9. Five-seed analysis: ResGRU-UNet vs AFNO V4
#
# Loads 5 seeds per finalist (42, 1337, 1, 2, 3), all trained with the same phase A
# recipe, and scores them on the same 214 validation windows with the competition
# metric. The measures and the decision rule were fixed before the seed runs
# (`reports/results.md`, 5-seed extension).
#
# - **Uncertainty:** 5-seed ensemble gap with a 95% block bootstrap (blocks of 13
#   consecutive windows = the 26 h a window spans, resampled within each month, 2,000
#   resamples); single-seed mean, standard deviation and the 25 pairwise comparisons.
# - **Where:** per lead hour, per month, per truth-concentration bin, episode vs other
#   pixels, per pixel (map) and one case study.
# - **Calibration:** V4's per-horizon scales, in sample and leave-one-month-out.
# - **Test set (no labels):** seed spread and agreement between the families.
# - **Cost:** parameters, s/epoch, inference time per window, peak GPU memory.
#
# Outputs: `analysis.json`, `fig_horizon.png`, `fig_bins.png`, `fig_maps.png`,
# `fig_case.png`, and for submission `preds_<family>_5seed.npy`,
# `preds_<family>_5seed_cal.npy`, `preds_resgru_s1337.npy`.

# %%
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm

SEEDS5 = [42, 1337, 1, 2, 3]
FAMS = ('afnov4', 'resgru')
NAMES = {'afnov4': 'AFNO V4', 'resgru': 'ResGRU-UNet'}
INPUT_ROOT = '/kaggle/input' if C.IS_KAGGLE else None
BLOCK, N_BOOT = 13, 2000
BINS = [0, 10, 25, 50, 100, 200, np.inf]
LEADS = [0, 5, 10, 15]
CAL_RANGE = (0.90, 1.10)
AB = 4


def ckpt_path(fam, seed):
    n = f's{seed}_phaseA_best.pt' if fam == 'afnov4' and seed in (42, 1337) else f'{fam}_s{seed}_phaseA_best.pt'
    d = find_dir(INPUT_ROOT, [n]) if INPUT_ROOT else None
    return os.path.join(d, n) if d else None


def make(fam):
    return AFNOv4() if fam == 'afnov4' else ResGRUUNet(C.C_IN)


def precheck():
    print('=' * 60, '\nPRE-CHECK: 10 checkpoints\n', '=' * 60)
    miss = []
    for f in FAMS:
        for s in SEEDS5:
            p = ckpt_path(f, s)
            ok = bool(p) and os.path.exists(p)
            print(f'  {"ok     " if ok else "MISSING"} {NAMES[f]} seed {s}: {p}')
            if not ok:
                miss.append((f, s))
    if miss:
        raise RuntimeError(f'missing checkpoints: {miss}')


def run_info(path, seed):
    d = os.path.dirname(path)
    for n in (f'summary_s{seed}.json', 'summary.json'):
        p = os.path.join(d, n)
        if os.path.exists(p):
            s = json.load(open(p))
            if s.get('seed') == seed:
                return s.get('secs_per_epoch'), os.path.join(d, f'preds_s{seed}.npy')
    return None, None


@torch.no_grad()
def val_arrays(model, val_ds):
    model.eval()
    vl = DataLoader(val_ds, batch_size=AB, shuffle=False, num_workers=C.WORKERS, pin_memory=True)
    N = len(val_ds)
    pred = np.zeros((N, C.T_OUT, C.H, C.W), np.float32)
    tgt = np.zeros_like(pred)
    msk = np.zeros((N, C.T_OUT, C.H, C.W), np.uint8)
    i = 0
    for x, t_log, t_raw, last_log, mask in vl:
        with torch.autocast('cuda', enabled=DEVICE.type == 'cuda'):
            pl = model(x.to(DEVICE), last_log.to(DEVICE))
        b = x.shape[0]
        pred[i:i + b] = F.relu(torch.expm1(pl.float().clamp(max=13.0))).cpu().numpy()
        tgt[i:i + b] = t_raw.numpy()
        msk[i:i + b] = mask.numpy().astype(np.uint8)
        i += b
    return pred, tgt, msk


@torch.no_grad()
def wh_terms(pred, tgt, msk):
    """Per window and lead hour: global SMAPE, episode SMAPE (nan: no episode), episode corr (nan: < 3 points)."""
    N = len(pred)
    g = np.zeros((N, C.T_OUT), np.float32)
    e = np.full_like(g, np.nan)
    c = np.full_like(g, np.nan)
    for i in range(0, N, AB):
        p = torch.from_numpy(pred[i:i + AB]).to(DEVICE)
        t = torch.from_numpy(tgt[i:i + AB]).to(DEVICE)
        m = torch.from_numpy(msk[i:i + AB].astype(np.float32)).to(DEVICE)
        sm = (p - t).abs() / (0.5 * (p.abs() + t.abs()) + EPS)
        n = m.sum(dim=(2, 3))
        g[i:i + AB] = sm.mean(dim=(2, 3)).cpu().numpy()
        es = (sm * m).sum(dim=(2, 3)) / n.clamp(min=1)
        e[i:i + AB] = torch.where(n > 0, es, torch.full_like(es, float('nan'))).cpu().numpy()
        corr, valid = _masked_corr(p, t, m, min_pts=3)
        c[i:i + AB] = torch.where(valid, corr, torch.full_like(corr, float('nan'))).cpu().numpy()
    return g, e, c


def score_t(g, e, c):
    return float((1 - np.nanmean(g) / 2 + (np.nanmean(c) + 1) / 2 + 1 - np.nanmean(e) / 2) / 3)


def terms_of(tr):
    g, e, c = tr
    return dict(score=round(score_t(g, e, c), 5), gsmape=round(float(np.nanmean(g)), 5),
                esmape=round(float(np.nanmean(e)), 5), ecorr=round(float(np.nanmean(c)), 5))


def fit_scales(pred, tgt, msk):
    """V4's per-horizon calibration (model/afno_v4.py fit_horizon_scales), without saving."""
    lo, hi = CAL_RANGE
    grid = np.linspace(lo, hi, 21)
    scales = np.ones(C.T_OUT, np.float32)
    for t in range(C.T_OUT):
        p, y, m = pred[:, t], tgt[:, t], msk[:, t].astype(bool)
        best, bs = -1e9, 1.0
        for s in grid:
            ps = p * s
            sm = np.abs(ps - y) / (0.5 * (np.abs(ps) + np.abs(y)) + EPS)
            g = sm.mean()
            e = sm[m].mean() if m.any() else 2.0
            obj = (1 - g / 2 + 1 - e / 2) / 2
            if obj > best:
                best, bs = obj, s
        scales[t] = bs
    return scales


@torch.no_grad()
def timing(model, val_ds, reps=20):
    xs = [val_ds[i] for i in range(AB)]
    x = torch.stack([a[0] for a in xs]).to(DEVICE)
    ll = torch.stack([a[3] for a in xs]).to(DEVICE)
    model.eval()
    if DEVICE.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    for _ in range(3):
        with torch.autocast('cuda', enabled=DEVICE.type == 'cuda'):
            model(x, ll)
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        with torch.autocast('cuda', enabled=DEVICE.type == 'cuda'):
            model(x, ll)
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    ms = (time.time() - t0) / reps / AB * 1000
    mem = torch.cuda.max_memory_allocated() / 2 ** 30 if DEVICE.type == 'cuda' else None
    return dict(ms_per_window=round(ms, 2), peak_gb_batch4=round(mem, 2) if mem else None,
                params=sum(p.numel() for p in model.parameters()))


def figures(res, tgt, ens, msk, w):
    col = {'afnov4': 'tab:blue', 'resgru': 'tab:orange'}
    hrs = np.arange(1, C.T_OUT + 1)
    fig, ax = plt.subplots(1, 4, figsize=(18, 3.6))
    for f in FAMS:
        for k, key in enumerate(('gsmape', 'esmape', 'ecorr', 'score')):
            for s in SEEDS5:
                ax[k].plot(hrs, res['per_hour'][f]['seeds'][str(s)][key], color=col[f], lw=0.6, alpha=0.4)
            ax[k].plot(hrs, res['per_hour'][f]['ensemble'][key], color=col[f], lw=2, label=NAMES[f])
            ax[k].set_title(key); ax[k].set_xlabel('lead hour')
    ax[0].legend()
    fig.tight_layout(); fig.savefig(os.path.join(C.WORK, 'fig_horizon.png'), dpi=120); plt.close(fig)

    lab = [f'{lo:g}-{hi:g}' for lo, hi in zip(BINS[:-1], BINS[1:])]
    xb = np.arange(len(lab))
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    for j, f in enumerate(FAMS):
        b = res['bins'][f]
        ax[0].bar(xb + 0.4 * j - 0.2, [r['smape'] for r in b], 0.4, color=col[f], label=NAMES[f])
        ax[1].bar(xb + 0.4 * j - 0.2, [r['log_bias'] for r in b], 0.4, color=col[f], label=NAMES[f])
    for a, t in zip(ax, ('mean SMAPE by truth bin (ug/m3)', 'mean log1p bias by truth bin')):
        a.set_xticks(xb); a.set_xticklabels(lab); a.set_title(t)
    ax[1].axhline(0, color='k', lw=0.5); ax[0].legend()
    fig.tight_layout(); fig.savefig(os.path.join(C.WORK, 'fig_bins.png'), dpi=120); plt.close(fig)

    maps = {f: res['_maps'][f] for f in FAMS}
    d = maps['resgru'] - maps['afnov4']
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    vmax = max(m.max() for m in maps.values())
    for a, f in zip(ax, FAMS):
        im = a.imshow(maps[f], origin='lower', vmin=0, vmax=vmax); a.set_title(f'{NAMES[f]}: mean SMAPE')
        fig.colorbar(im, ax=a, shrink=0.8)
    lim = float(np.abs(d).max()) or 1e-6
    im = ax[2].imshow(d, origin='lower', cmap='RdBu_r', norm=TwoSlopeNorm(0, -lim, lim))
    ax[2].set_title('ResGRU-UNet minus AFNO V4 (blue: ResGRU-UNet lower)'); fig.colorbar(im, ax=ax[2], shrink=0.8)
    fig.tight_layout(); fig.savefig(os.path.join(C.WORK, 'fig_maps.png'), dpi=120); plt.close(fig)

    rows = [('truth', tgt[w])] + [(NAMES[f], ens[f][w]) for f in FAMS]
    vmin = max(0.5, min(float(np.percentile(r[1][LEADS], 1)) for r in rows))
    vmx = max(float(r[1][LEADS].max()) for r in rows)
    fig, ax = plt.subplots(3, len(LEADS), figsize=(4 * len(LEADS), 10))
    for i, (name, arr) in enumerate(rows):
        for j, t in enumerate(LEADS):
            im = ax[i, j].imshow(np.clip(arr[t], vmin, None), origin='lower', norm=LogNorm(vmin, vmx))
            ax[i, j].set_title(f'{name}, lead {t + 1} h'); ax[i, j].axis('off')
            if i == 0:
                ax[i, j].contour(msk[w, t], levels=[0.5], colors='w', linewidths=0.5)
    fig.colorbar(im, ax=ax, shrink=0.6, label='PM2.5 (ug/m3), white: episode pixels')
    fig.savefig(os.path.join(C.WORK, 'fig_case.png'), dpi=110); plt.close(fig)


def analyze():
    t_start = time.time()
    if C.IS_KAGGLE:
        check_gpu()
    precheck()
    stats = build_stats()
    months, static = load_months(stats)
    val_ds = PM25Windows(months, static, stats, 'val')
    test_ds = PM25Test(stats, static)
    mon = np.array([mi for mi, _ in val_ds.samples])
    keys = val_ds.keys
    res = dict(seeds=SEEDS5, n_val_windows=len(val_ds), months=keys)
    tgt = msk = None
    T, ens, tests, seeds, cost, checks = {}, {}, {f: [] for f in FAMS}, {f: {} for f in FAMS}, {}, {}

    print('=' * 60, '\nSTEP A: 10 models on val and test\n', '=' * 60)
    for f in FAMS:
        for s in SEEDS5:
            path = ckpt_path(f, s)
            ck = torch.load(path, map_location=DEVICE, weights_only=False)
            model = make(f).to(DEVICE)
            model.load_state_dict(ck['model'])
            vp, t, m = val_arrays(model, val_ds)
            if tgt is None:
                tgt, msk = t, m
            T[f, s] = wh_terms(vp, tgt, msk)
            ens[f] = vp / len(SEEDS5) if f not in ens else ens[f] + vp / len(SEEDS5)
            tp = predict(model, test_ds, AB)
            assert tp.shape == (C.N_TEST, C.H, C.W, C.T_OUT) and np.isfinite(tp).all()
            tests[f].append(tp)
            spe, own = run_info(path, s)
            if own and os.path.exists(own):
                ref = np.load(own)
                rel = float(np.abs(tp - ref).mean() / np.abs(ref).mean())
                checks[f'{f} s{s} test preds equal the training run output (rel < 1e-3)'] = rel < 1e-3
            logged = float(ck.get('metrics', {}).get('score', float('nan')))
            seeds[f][s] = dict(**terms_of(T[f, s]), logged=round(logged, 5), epoch=ck.get('epoch'),
                               secs_per_epoch=spe)
            checks[f'{f} s{s} val equals logged'] = abs(seeds[f][s]['score'] - logged) < 1e-4
            if s == SEEDS5[0]:
                cost[f] = timing(model, val_ds)
            print(f'  {NAMES[f]} seed {s}: val {seeds[f][s]["score"]:.5f} (logged {logged:.5f})')
            del model, vp
            gc.collect(); torch.cuda.empty_cache()
    del months
    gc.collect()

    print('=' * 60, '\nSTEP B: ensembles, uncertainty, pairwise\n', '=' * 60)
    TE = {f: wh_terms(ens[f], tgt, msk) for f in FAMS}
    res['seeds_val'] = seeds
    res['ensemble_val'] = {f: terms_of(TE[f]) for f in FAMS}
    sc = {f: np.array([seeds[f][s]['score'] for s in SEEDS5]) for f in FAMS}
    res['single_seed_val'] = {f: dict(mean=round(float(sc[f].mean()), 5), sd=round(float(sc[f].std(ddof=1)), 5),
                                      min=round(float(sc[f].min()), 5), max=round(float(sc[f].max()), 5)) for f in FAMS}
    res['pairs_resgru_wins'] = int(sum(a > b for a in sc['resgru'] for b in sc['afnov4']))
    rng = np.random.default_rng(0)
    blocks = []
    for mi in range(len(keys)):
        idx = np.nonzero(mon == mi)[0]
        blocks.append([idx[j:j + BLOCK] for j in range(0, len(idx), BLOCK)])
    be, bm = [], []
    for _ in range(N_BOOT):
        ix = np.concatenate([np.concatenate([b[k] for k in rng.integers(0, len(b), len(b))]) for b in blocks])
        sel = lambda tr: tuple(a[ix] for a in tr)
        be.append(score_t(*sel(TE['resgru'])) - score_t(*sel(TE['afnov4'])))
        bm.append(np.mean([score_t(*sel(T['resgru', s])) for s in SEEDS5])
                  - np.mean([score_t(*sel(T['afnov4', s])) for s in SEEDS5]))
    gap_e = res['ensemble_val']['resgru']['score'] - res['ensemble_val']['afnov4']['score']
    gap_m = res['single_seed_val']['resgru']['mean'] - res['single_seed_val']['afnov4']['mean']
    ci = lambda v: [round(float(np.percentile(v, 2.5)), 5), round(float(np.percentile(v, 97.5)), 5)]
    res['gap'] = dict(ensemble=round(gap_e, 5), ensemble_ci95=ci(be), ensemble_boot_share_above_0=round(float(np.mean(np.array(be) > 0)), 4),
                      single_seed_mean=round(gap_m, 5), single_seed_mean_ci95=ci(bm),
                      block=BLOCK, n_boot=N_BOOT, blocks_per_month=[len(b) for b in blocks])
    print('  gap:', res['gap'])

    print('=' * 60, '\nSTEP C: where the difference is\n', '=' * 60)
    hour = lambda tr: dict(gsmape=np.nanmean(tr[0], 0), esmape=np.nanmean(tr[1], 0), ecorr=np.nanmean(tr[2], 0))
    def ph(tr):
        h = hour(tr)
        h['score'] = (1 - h['gsmape'] / 2 + (h['ecorr'] + 1) / 2 + 1 - h['esmape'] / 2) / 3
        return {k: [round(float(v), 5) for v in a] for k, a in h.items()}
    res['per_hour'] = {f: dict(ensemble=ph(TE[f]), seeds={str(s): ph(T[f, s]) for s in SEEDS5}) for f in FAMS}
    res['per_month'] = {f: {keys[mi]: round(score_t(*(a[mon == mi] for a in TE[f])), 5) for mi in range(len(keys))} for f in FAMS}
    res['bins'], res['episode_split'], res['_maps'] = {}, {}, {}
    ep = msk.astype(bool)
    for f in FAMS:
        sm = np.abs(ens[f] - tgt) / (0.5 * (np.abs(ens[f]) + np.abs(tgt)) + EPS)
        lb = np.log1p(ens[f]) - np.log1p(tgt)
        rows = []
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            s_ = (tgt >= lo) & (tgt < hi)
            rows.append(dict(lo=lo, hi=None if np.isinf(hi) else hi, share=round(float(s_.mean()), 5),
                             smape=round(float(sm[s_].mean()), 5) if s_.any() else None,
                             log_bias=round(float(lb[s_].mean()), 5) if s_.any() else None))
        res['bins'][f] = rows
        res['episode_split'][f] = {k: dict(smape=round(float(sm[s_].mean()), 5), log_bias=round(float(lb[s_].mean()), 5))
                                   for k, s_ in (('episode', ep), ('other', ~ep))}
        res['_maps'][f] = sm.mean(axis=(0, 1))
        del sm, lb
    cell = tgt.mean(axis=(0, 1))
    q = np.quantile(cell, np.linspace(0, 1, 11))
    dec = np.clip(np.searchsorted(q, cell, side='right') - 1, 0, 9)
    res['cell_deciles'] = [dict(decile=k + 1, cell_mean_lo=round(float(q[k]), 3), cell_mean_hi=round(float(q[k + 1]), 3),
                                **{f: round(float(res['_maps'][f][dec == k].mean()), 5) for f in FAMS}) for k in range(10)]
    dmap = res['_maps']['resgru'] - res['_maps']['afnov4']
    res['cells_resgru_lower_smape'] = round(float((dmap < 0).mean()), 4)
    w = int(msk.reshape(len(msk), -1).sum(1).argmax())
    mi, s0 = val_ds.samples[w]
    res['case'] = dict(window=w, month=keys[mi], start=int(s0), episode_pixels=int(msk[w].sum()),
                       **{f: terms_of(tuple(a[w:w + 1] for a in TE[f])) for f in FAMS})

    print('=' * 60, '\nSTEP D: calibration (in sample, leave one month out)\n', '=' * 60)
    res['calibration'], scales_in = {}, {}
    for f in FAMS:
        s_in = fit_scales(ens[f], tgt, msk)
        scales_in[f] = s_in
        cal = np.empty_like(ens[f])
        lomo = {}
        for mi in range(len(keys)):
            tr, te = mon != mi, mon == mi
            s_m = fit_scales(ens[f][tr], tgt[tr], msk[tr])
            cal[te] = ens[f][te] * s_m[None, :, None, None]
            lomo[keys[mi]] = [round(float(v), 3) for v in s_m]
        res['calibration'][f] = dict(
            scales_in_sample=[round(float(v), 3) for v in s_in],
            score_in_sample=terms_of(wh_terms(ens[f] * s_in[None, :, None, None], tgt, msk)),
            score_lomo=terms_of(wh_terms(cal, tgt, msk)), scales_lomo=lomo)
        del cal
        print(f'  {NAMES[f]}: uncalibrated {res["ensemble_val"][f]["score"]:.5f}, in sample '
              f'{res["calibration"][f]["score_in_sample"]["score"]:.5f}, leave-one-month-out '
              f'{res["calibration"][f]["score_lomo"]["score"]:.5f}')

    print('=' * 60, '\nSTEP E: test predictions, seed spread, agreement\n', '=' * 60)
    et = {}
    for f in FAMS:
        st = np.stack(tests[f])
        et[f] = st.mean(0)
        res.setdefault('test', {})[f] = dict(seed_cv=round(float(st.std(0).mean() / st.mean()), 5))
        for tag, arr in ((f'preds_{f}_5seed.npy', et[f]), (f'preds_{f}_5seed_cal.npy', et[f] * scales_in[f][None, None, None, :])):
            arr = arr.astype(np.float32)
            assert arr.shape == (C.N_TEST, C.H, C.W, C.T_OUT) and np.isfinite(arr).all()
            np.save(os.path.join(C.WORK, tag), arr)
        del st
    np.save(os.path.join(C.WORK, 'preds_resgru_s1337.npy'), tests['resgru'][SEEDS5.index(1337)].astype(np.float32))
    la, lr = np.log1p(et['afnov4']).ravel(), np.log1p(et['resgru']).ravel()
    res['test']['agreement'] = dict(corr_log1p=round(float(np.corrcoef(la, lr)[0, 1]), 5),
                                    mean_abs_log_ratio=round(float(np.abs(lr - la).mean()), 5))

    res['cost'] = cost
    res['checks'] = checks
    res['all_checks_pass'] = all(checks.values())
    figures(res, tgt, ens, msk, w)
    res.pop('_maps')
    res['minutes'] = round((time.time() - t_start) / 60, 1)
    json.dump(res, open(os.path.join(C.WORK, 'analysis.json'), 'w'), indent=1)
    print('  analysis.json and 4 figures written; checks:', 'all pass' if res['all_checks_pass'] else checks)

# %% [markdown]
# ## 11. Regression metrics for every compared model: MAE, RMSE, R²
#
# Everything else in the study uses the competition metric (global SMAPE, episode SMAPE and
# the episode Pearson correlation r). This section adds standard regression metrics for every
# model the study compared, on the same 214 validation windows. No training happens here.
#
# - round one (1 seed): AFNO V4, ResGRU-UNet, FMG-WACLU, ConvLSTM, patch AFNO
# - final round (2 seeds): 2-seed means of AFNO V4 and ResGRU-UNet, plain and calibrated
# - 5 seeds: the 10 single seeds; 5-seed means plain, calibrated in sample and leave-one-month-out
# - ablations (1 seed): allskip, pad, nosparse, norec
# - persistence: the last input hour repeated for 16 h
#
# Definitions, in ug/m3, pooled over every val value (214 windows x 16 lead hours x 140 x 124):
# MAE = mean |p - y|; RMSE = sqrt(mean (p - y)^2); R² = 1 - sum (p - y)^2 / sum (y - mean y)^2.
# The same on log1p values; per lead hour (R² against that hour's own mean); and on episode
# and other pixels. r_pooled is the Pearson correlation over all values (r² equals R² only for
# a least-squares fit). Test labels are hidden, so none of this exists for the test set.
# Windows are 26 h long with a 2 h stride, so most target pixel-hours enter several windows.

# %%
ROUND1 = ['afnov4', 'resgru', 'fmgwaclu', 'convlstm', 'patchafno']
ABL = ['afnov4_allskip', 'afnov4_pad', 'afnov4_nosparse', 'resgru_norec']
LABEL = {'afnov4': 'AFNO V4', 'resgru': 'ResGRU-UNet', 'fmgwaclu': 'FMG-WACLU', 'convlstm': 'ConvLSTM',
         'patchafno': 'patch AFNO', 'afnov4_allskip': 'AFNO V4, all-hour skips (H1)',
         'afnov4_pad': 'AFNO V4, padded (H2)', 'afnov4_nosparse': 'AFNO V4, no soft-threshold (H3)',
         'resgru_norec': 'ResGRU-UNet, no recurrence (H4)'}
PERSIST_REF = 0.8515


def ckpt_of(bb, seed):
    n = f's{seed}_phaseA_best.pt' if bb == 'afnov4' and seed in (42, 1337) else f'{bb}_s{seed}_phaseA_best.pt'
    d = find_dir(INPUT_ROOT, [n]) if INPUT_ROOT else None
    return os.path.join(d, n) if d else None


def runs_needed():
    return ([(f, s) for f in FAMS for s in SEEDS5] + [(bb, 42) for bb in ROUND1 if bb not in FAMS]
            + [(bb, 42) for bb in ABL])


def stage_of(bb, s):
    if bb in ABL:
        return 'ablation (1 seed)'
    if bb in FAMS:
        return 'round one (1 seed); 5 seeds' if s == 42 else '5 seeds'
    return 'round one (1 seed)'


def find_ref(name, key=None):
    if not INPUT_ROOT:
        return None
    for dp, _, fs in os.walk(INPUT_ROOT):
        if name in fs:
            p = os.path.join(dp, name)
            if key is None or key in json.load(open(p)):
                return p
    return None


def truth_stats(tgt, msk):
    y = tgt.astype(np.float64)
    ep = msk.astype(bool)
    ly = np.log1p(y)
    sst = lambda a: float(((a - a.mean()) ** 2).sum())
    return dict(n=y.size, sy=float(y.sum()), sy2=float((y * y).sum()), mean=float(y.mean()),
                sst=sst(y), lsst=sst(ly), sst_h=[sst(y[:, h]) for h in range(C.T_OUT)],
                sst_ep=sst(y[ep]), sst_ot=sst(y[~ep]))


def reg_metrics(pred, tgt, msk, ts):
    N = ts['n']
    ph = dict(mae=[], rmse=[], r2=[])
    sae = sse = lsae = lsse = sp = sp2 = spy = 0.0
    ep = dict(sae=0.0, sse=0.0, n=0)
    ot = dict(sae=0.0, sse=0.0, n=0)
    for h in range(C.T_OUT):
        p = pred[:, h].astype(np.float64)
        y = tgt[:, h].astype(np.float64)
        m = msk[:, h].astype(bool)
        d = p - y
        a, s2 = float(np.abs(d).sum()), float((d * d).sum())
        ph['mae'].append(round(a / d.size, 4))
        ph['rmse'].append(round(float(np.sqrt(s2 / d.size)), 4))
        ph['r2'].append(round(1 - s2 / ts['sst_h'][h], 5))
        sae += a
        sse += s2
        ld = np.log1p(p) - np.log1p(y)
        lsae += float(np.abs(ld).sum())
        lsse += float((ld * ld).sum())
        sp += float(p.sum())
        sp2 += float((p * p).sum())
        spy += float((p * y).sum())
        for acc, sel in ((ep, m), (ot, ~m)):
            ds = d[sel]
            acc['sae'] += float(np.abs(ds).sum())
            acc['sse'] += float((ds * ds).sum())
            acc['n'] += int(ds.size)
    cov = spy - sp * ts['sy'] / N
    r = cov / np.sqrt(max(sp2 - sp * sp / N, 1e-30) * (ts['sy2'] - ts['sy'] ** 2 / N))
    sub = lambda acc, sst: dict(mae=round(acc['sae'] / max(acc['n'], 1), 4),
                                rmse=round(float(np.sqrt(acc['sse'] / max(acc['n'], 1))), 4),
                                r2=round(1 - acc['sse'] / sst, 5), n=acc['n'])
    return dict(mae=round(sae / N, 4), rmse=round(float(np.sqrt(sse / N)), 4), r2=round(1 - sse / ts['sst'], 5),
                mae_log1p=round(lsae / N, 5), rmse_log1p=round(float(np.sqrt(lsse / N)), 5),
                r2_log1p=round(1 - lsse / ts['lsst'], 5), r_pooled=round(float(r), 5),
                per_hour=ph, episode=sub(ep, ts['sst_ep']), other=sub(ot, ts['sst_ot']))


@torch.no_grad()
def val_persistence(val_ds):
    vl = DataLoader(val_ds, batch_size=AB, shuffle=False, num_workers=C.WORKERS)
    out = np.zeros((len(val_ds), C.T_OUT, C.H, C.W), np.float32)
    i = 0
    for x, t_log, t_raw, last_log, mask in vl:
        p = F.relu(torch.expm1(last_log.float().clamp(max=13.0)))
        out[i:i + len(p)] = p[:, None].expand(-1, C.T_OUT, -1, -1).numpy()
        i += len(p)
    return out


def calibrated(e, tgt, msk, mon, n_months):
    s_in = fit_scales(e, tgt, msk)
    lomo = np.empty_like(e)
    for mi in range(n_months):
        tr, te = mon != mi, mon == mi
        lomo[te] = e[te] * fit_scales(e[tr], tgt[tr], msk[tr])[None, :, None, None]
    return e * s_in[None, :, None, None], lomo, s_in


def regression():
    t0 = time.time()
    if C.IS_KAGGLE:
        check_gpu()
    need = runs_needed()
    miss = [(b, s) for b, s in need if not ckpt_of(b, s)]
    ana, fin = find_ref('analysis.json'), find_ref('summary.json', 'final_round')
    print('=' * 60, f'\nPRE-CHECK: {len(need)} checkpoints, analysis.json, final-round summary\n', '=' * 60)
    for b, s in need:
        print(f'  {"MISSING" if (b, s) in miss else "ok     "} {b} seed {s}')
    if miss or not ana or not fin:
        raise RuntimeError(f'missing: checkpoints {miss}, analysis.json {bool(ana)}, final summary {bool(fin)}')
    stats = build_stats()
    months, static = load_months(stats)
    val_ds = PM25Windows(months, static, stats, 'val')
    mon = np.array([mi for mi, _ in val_ds.samples])
    keys = val_ds.keys
    res = dict(n_val_windows=len(val_ds), months=keys, models={}, checks={},
               definitions='ug/m3, pooled over 214 val windows x 16 h x 140 x 124; R2 = 1 - SSE/SST; '
                           'per-hour R2 uses each hour\'s own mean; r_pooled = Pearson r over all values')
    tgt = msk = ts = None
    e2 = {f: None for f in FAMS}
    e5 = {f: None for f in FAMS}

    def record(key, label, stage, pred, **extra):
        m = reg_metrics(pred, tgt, msk, ts)
        m.update(terms_of(wh_terms(pred, tgt, msk)))
        m.update(label=label, stage=stage, **extra)
        res['models'][key] = m
        print(f'  {label:42s} {stage:28s} MAE {m["mae"]:8.3f}  RMSE {m["rmse"]:8.3f}  R2 {m["r2"]:.4f}  '
              f'R2 log {m["r2_log1p"]:.4f}  score {m["score"]:.5f}')

    print('=' * 60, '\nSTEP A: every trained model on val\n', '=' * 60)
    for bb, s in need:
        ck = torch.load(ckpt_of(bb, s), map_location=DEVICE, weights_only=False)
        model = BACKBONES[bb]().to(DEVICE)
        model.load_state_dict(ck['model'])
        vp, t, m = val_arrays(model, val_ds)
        if tgt is None:
            tgt, msk = t, m
            ts = truth_stats(tgt, msk)
            res['truth'] = dict(n=ts['n'], mean=round(ts['mean'], 4), sd=round(float(np.sqrt(ts['sst'] / ts['n'])), 4))
        logged = float(ck.get('metrics', {}).get('score', float('nan')))
        record(f'{bb}_s{s}', f'{LABEL[bb]} seed {s}', stage_of(bb, s), vp, seed=s, epoch=ck.get('epoch'),
               logged=round(logged, 5))
        res['checks'][f'{bb} s{s} val equals logged'] = abs(res['models'][f'{bb}_s{s}']['score'] - logged) < 1e-4
        if bb in FAMS:
            e5[bb] = vp / 5 if e5[bb] is None else e5[bb] + vp / 5
            if s in (42, 1337):
                e2[bb] = vp / 2 if e2[bb] is None else e2[bb] + vp / 2
        del model, vp
        gc.collect()
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
    del months
    gc.collect()

    print('=' * 60, '\nSTEP B: 2- and 5-seed means, calibrated\n', '=' * 60)
    for f in FAMS:
        for tag, e, stage in (('2seed', e2[f], 'final round (2 seeds)'), ('5seed', e5[f], '5 seeds')):
            n = tag[0]
            record(f'{f}_{tag}', f'{LABEL[f]} {n}-seed mean', stage, e)
            cin, clomo, s_in = calibrated(e, tgt, msk, mon, len(keys))
            record(f'{f}_{tag}_cal_in', f'{LABEL[f]} {n}-seed + cal (in sample)', stage, cin,
                   scales=[round(float(v), 3) for v in s_in])
            record(f'{f}_{tag}_cal_lomo', f'{LABEL[f]} {n}-seed + cal (leave-one-month-out)', stage, clomo)
            del cin, clomo
    del e2, e5
    gc.collect()

    print('=' * 60, '\nSTEP C: persistence\n', '=' * 60)
    record('persistence', 'persistence (last input hour)', 'baseline', val_persistence(val_ds))

    M = res['models']
    res['single_seed_summary'] = {}
    for f in FAMS:
        rows = [M[f'{f}_s{s}'] for s in SEEDS5]
        res['single_seed_summary'][f] = {k: dict(mean=round(float(np.mean([r[k] for r in rows])), 5),
                                                 sd=round(float(np.std([r[k] for r in rows], ddof=1)), 5))
                                         for k in ('mae', 'rmse', 'r2', 'r2_log1p', 'score')}
    A = json.load(open(ana))
    FR = json.load(open(fin))['final_round']
    ck = res['checks']
    for f in FAMS:
        ck[f'{f} 5-seed mean score equals analysis.json'] = abs(M[f'{f}_5seed']['score'] - A['ensemble_val'][f]['score']) < 1e-4
        ck[f'{f} 5-seed calibrated in sample equals analysis.json'] = abs(
            M[f'{f}_5seed_cal_in']['score'] - A['calibration'][f]['score_in_sample']['score']) < 1e-4
        ck[f'{f} 5-seed calibrated leave-one-month-out equals analysis.json'] = abs(
            M[f'{f}_5seed_cal_lomo']['score'] - A['calibration'][f]['score_lomo']['score']) < 1e-4
        ck[f'{f} 2-seed mean score equals the final round'] = abs(M[f'{f}_2seed']['score'] - FR[f]['two_seed']['score']) < 1e-4
        ck[f'{f} 2-seed calibrated equals the final round'] = abs(
            M[f'{f}_2seed_cal_in']['score'] - FR[f]['two_seed_cal_in_sample']['score']) < 1e-4
    if PERSIST_REF is not None:
        ck['persistence score equals 0.8515 (t8) within 0.001'] = abs(M['persistence']['score'] - PERSIST_REF) < 1e-3
    self_fit = reg_metrics(tgt, tgt, msk, ts)
    ck['truth against itself: MAE 0, RMSE 0, R2 1'] = self_fit['mae'] == 0 and self_fit['rmse'] == 0 and self_fit['r2'] == 1
    res['all_checks_pass'] = all(ck.values())
    res['minutes'] = round((time.time() - t0) / 60, 1)
    json.dump(res, open(os.path.join(C.WORK, 'regression_metrics.json'), 'w'), indent=1)
    print(f'  regression_metrics.json written ({len(M)} models); checks:',
          'all pass' if res['all_checks_pass'] else {k: v for k, v in ck.items() if not v})

# %% [markdown]
# ## 12. Baselines: persistence and seasonal naive (same hour, 24 h earlier)
#
# Seasonal naive predicts each target hour with the value 24 h earlier at the same pixel.
# For lead hours 1-14 that hour lies before the 10-hour input window, so it is read from the
# month series. It cannot be computed on the test set (10 input hours only).
# Same 214 val windows and definitions as section 11. CPU only, no model.

# %%
@torch.no_grad()
def val_truth(val_ds):
    vl = DataLoader(val_ds, batch_size=AB, shuffle=False, num_workers=C.WORKERS)
    shape = (len(val_ds), C.T_OUT, C.H, C.W)
    tgt, per, msk = np.zeros(shape, np.float32), np.zeros(shape, np.float32), np.zeros(shape, np.uint8)
    i = 0
    for x, t_log, t_raw, last_log, mask in vl:
        b = len(t_raw)
        tgt[i:i + b] = t_raw.numpy()
        msk[i:i + b] = mask.numpy().astype(np.uint8)
        per[i:i + b] = F.relu(torch.expm1(last_log.float().clamp(max=13.0)))[:, None].expand(-1, C.T_OUT, -1, -1).numpy()
        i += b
    return tgt, msk, per


def seasonal_naive(val_ds, raw):
    out = np.zeros((len(val_ds), C.T_OUT, C.H, C.W), np.float32)
    first = []
    for w, (mi, s0) in enumerate(val_ds.samples):
        lo = s0 + C.T_IN - 24
        first.append(lo)
        out[w] = raw[val_ds.keys[mi]][lo:lo + C.T_OUT]
    return out, min(first)


def baselines():
    t0 = time.time()
    ref = find_ref('regression_metrics.json')
    if not ref:
        raise RuntimeError('regression_metrics.json (regression-metrics kernel output) is not attached')
    stats = build_stats()
    months, static = load_months(stats)
    val_ds = PM25Windows(months, static, stats, 'val')
    del months
    tgt, msk, per = val_truth(val_ds)
    ts = truth_stats(tgt, msk)
    raw = {k: np.load(os.path.join(C.RAW, k, 'cpm25.npy'), mmap_mode='r') for k in val_ds.keys}
    sn, lo = seasonal_naive(val_ds, raw)
    ck = {'24 h earlier lies inside the month for every window': lo >= 0,
          'loader truth equals the raw series at the target hours': all(
              np.allclose(tgt[w], raw[val_ds.keys[mi]][s0 + C.T_IN:s0 + C.T_IN + C.T_OUT], rtol=1e-4, atol=1e-4)
              for w, (mi, s0) in enumerate(val_ds.samples))}
    res = dict(n_val_windows=len(val_ds), models={}, checks=ck)
    for key, label, p in (('persistence', 'persistence (last input hour)', per),
                          ('seasonal_naive', 'seasonal naive (same hour, 24 h earlier)', sn)):
        m = reg_metrics(p, tgt, msk, ts)
        m.update(terms_of(wh_terms(p, tgt, msk)))
        m['label'] = label
        res['models'][key] = m
        print(f'  {label:42s} MAE {m["mae"]:.3f}  RMSE {m["rmse"]:.3f}  R2 {m["r2"]:.4f}  score {m["score"]:.5f}')
    R = json.load(open(ref))['models']['persistence']
    ck['persistence equals the regression run'] = all(abs(res['models']['persistence'][k] - R[k]) < 1e-3
                                                     for k in ('mae', 'rmse', 'r2', 'score'))
    res['all_checks_pass'] = all(ck.values())
    res['minutes'] = round((time.time() - t0) / 60, 1)
    json.dump(res, open(os.path.join(C.WORK, 'baselines.json'), 'w'), indent=1)
    print('  baselines.json written; checks:', 'all pass' if res['all_checks_pass'] else ck)

# %% [markdown]
# ## 14. Per-pixel maps for the paper
#
# Aggregate maps over the 214 val windows and 16 lead hours (time-averaged, 140 x 124 each):
# truth mean, episode frequency, and per-pixel MAE, mean SMAPE and R² (over time) for
# persistence and the 5-seed means of AFNO V4 and ResGRU-UNet, plus the grid's lat/lon.
# Output: maps.npz (no raw fields).

# %%
def pixel_maps(pred, tgt):
    p, y = pred.astype(np.float64), tgt.astype(np.float64)
    d = p - y
    n = y.shape[0] * y.shape[1]
    sse = (d * d).sum((0, 1))
    sst = ((y - y.mean((0, 1))) ** 2).sum((0, 1))
    smape = (np.abs(d) / (0.5 * (np.abs(p) + np.abs(y)) + EPS)).mean((0, 1))
    return dict(mae=np.abs(d).sum((0, 1)) / n, smape=smape, r2=1 - sse / np.maximum(sst, 1e-12))


def maps():
    t0 = time.time()
    if C.IS_KAGGLE:
        check_gpu()
    miss = [(f, s) for f in FAMS for s in SEEDS5 if not ckpt_of(f, s)]
    if miss:
        raise RuntimeError(f'missing checkpoints {miss}')
    stats = build_stats()
    months, static = load_months(stats)
    val_ds = PM25Windows(months, static, stats, 'val')
    del months
    tgt = msk = None
    ens = {}
    for f in FAMS:
        for s in SEEDS5:
            model = BACKBONES[f]().to(DEVICE)
            model.load_state_dict(torch.load(ckpt_of(f, s), map_location=DEVICE, weights_only=False)['model'])
            vp, t, m = val_arrays(model, val_ds)
            if tgt is None:
                tgt, msk = t, m
            ens[f] = vp / 5 if f not in ens else ens[f] + vp / 5
            del model, vp
            gc.collect()
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
    out = dict(lat=stats['lat'], lon=stats['lon'], truth_mean=tgt.mean((0, 1)), episode_freq=msk.mean((0, 1)))
    for key, pred in (('persistence', val_persistence(val_ds)), ('afnov4', ens['afnov4']), ('resgru', ens['resgru'])):
        for k, v in pixel_maps(pred, tgt).items():
            out[f'{key}_{k}'] = v
    out = {k: np.asarray(v, np.float32) for k, v in out.items()}
    ck = {'every map is 140 x 124 and finite': all(v.shape == (C.H, C.W) and np.isfinite(v).all() for v in out.values()),
          'ensemble MAE map averages to the regression run (6.32 / 6.99 within 0.01)':
              abs(float(out['resgru_mae'].mean()) - 6.3238) < 0.01 and abs(float(out['afnov4_mae'].mean()) - 6.9877) < 0.01}
    np.savez_compressed(os.path.join(C.WORK, 'maps.npz'), **out)
    json.dump(dict(checks=ck, all_checks_pass=all(ck.values()), keys=sorted(out), minutes=round((time.time() - t0) / 60, 1)),
              open(os.path.join(C.WORK, 'maps.json'), 'w'), indent=1)
    print('  maps.npz written; checks:', ck)


if __name__ == '__main__':
    maps()
