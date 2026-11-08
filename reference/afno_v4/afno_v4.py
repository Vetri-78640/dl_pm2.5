# %% [markdown]
# # Code4CleanAir — Model 4: AFNO v4 (research edition)
#
# Builds on the reviewed v3 pipeline (private 0.8609, val 0.9030 -> measured
# val-to-private gap ~0.042). Changes are literature-grounded:
#
# **Feature engineering (atmospheric physics):**
# 1. Ventilation coefficient = PBLH x wind speed - the canonical dispersion
#    predictor in air-quality meteorology (low ventilation -> accumulation
#    -> episodes). Garratt 1992; Sao Paulo/China stagnation studies.
# 2. Advection tendency -(u dC/dx + v dC/dy) of log PM2.5 - first-order
#    chemistry-transport surrogate; tells the model what transport is
#    arriving rather than asking it to discover differentiation.
# 3. PM tendency dC/dt within the window (consistent train/test: first
#    frame 0) - distinguishes building vs decaying episodes.
# 4. Diurnal climatology anomaly - log PM2.5 minus per-pixel hour-of-day
#    train climatology, in per-pixel std units. Mirrors the STL episode
#    definition's "deviation from expected" structure.
#
# **Techniques (AdaptFNO NeurIPS-ML4PS 2025, in this repo; SimVP; FourCastNet):**
# 5. Multiscale spectral mixing: one AFNO block at 35x31 (encoder level 2)
#    in addition to the 18x16 bottleneck stack - higher-frequency transport
#    modes than the bottleneck can represent.
# 6. Horizon-weighted loss: linear ramp 0.7 -> 1.3 over h+1..h+16. h+1 is
#    nearly persistence (SMAPE 0.07); h+16 is where the score is won
#    (SMAPE 0.40). AdaptFNO weights early steps for rollout stability; the
#    direct multi-horizon setting inverts that argument.
# 7. Seed ensemble (SEEDS config) - independent inits averaged at inference.
# 8. Per-horizon calibration on validation: a scalar multiplier per output
#    hour fitted to maximize (NormGlobalSMAPE + NormEpisodeSMAPE)/2.
#    Pearson correlation is invariant to positive scaling, so NormEpisodeCorr
#    is untouched by construction.
#
# All v3 review fixes are retained (leak-free split, official STL episode
# masks, hour estimation, std floors, input clipping, crash-proof inference,
# resume markers, EMA ramp, CSV epoch logging, safety preds.npy).

# %%
import os, gc, math, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from collections import defaultdict
import multiprocessing as mp

try:
    MP_CTX = mp.get_context('fork')
except ValueError:
    MP_CTX = mp.get_context()

T0_GLOBAL = time.time()


class C:
    IS_KAGGLE = os.path.exists('/kaggle/input')
    DATA_ROOT = ('/kaggle/input/competitions/anrf-aise-hack-phase-2-theme-2-'
                 'pollution-forecasting-iitd/aisehack-theme-2'
                 if IS_KAGGLE else 'Dataset')
    WORK = '/kaggle/working' if IS_KAGGLE else './work'
    # previous session's notebook output (attach as dataset) for resume
    PREV = None

    RAW = os.path.join(DATA_ROOT, 'raw')
    TEST = os.path.join(DATA_ROOT, 'test_in')

    MONTHS = ['APRIL_16', 'JULY_16', 'OCT_16', 'DEC_16']
    FEATS = ['cpm25', 'q2', 't2', 'u10', 'v10', 'pblh', 'psfc', 'swdown', 'rain']
    LOG_FEATS = {'cpm25', 'rain'}
    # precomputed dynamic channels: 9 met + ws + vent + adv
    N_PRE = 12
    H, W = 140, 124
    T_IN, T_OUT = 10, 16
    WIN = 26
    N_TEST = 218
    # 12 precomputed + dpm + hour_sin + hour_cos + clim_anom + 4 static
    C_IN = 20

    VAL_FRAC = 0.18
    GAP = 12
    VAL_STRIDE = 2

    STL_ROBUST = True
    STL_PROCS = 4

    ENC = [64, 128, 256]
    AFNO_BLOCKS = 6
    MID_AFNO_BLOCKS = 1    # multiscale: AFNO at 35x31 (AdaptFNO-style)
    POST_AFNO_BLOCKS = 2
    AFNO_HEADS = 8
    MLP_RATIO = 2.0
    DROP = 0.1
    TRANS_LAYERS = 4
    CLIP_IN = 10.0

    SEEDS = [42, 1337]     # seed ensemble; use [42] if time is short

    A_EPOCHS, A_LR, A_BATCH, A_PATIENCE = 16, 2.5e-4, 4, 16
    B_EPOCHS, B_LR, B_BATCH, B_PATIENCE = 6, 5e-5, 4, 3
    EP_WT_A, EP_WT_B = 6.0, 10.0
    LOSS_A = dict(huber=0.40, gsmape=0.25, esmape=0.15, ecorr=0.20)
    LOSS_B = dict(huber=0.25, gsmape=0.20, esmape=0.25, ecorr=0.30)
    HW_MIN, HW_MAX = 0.7, 1.3   # horizon loss weights, h+1 -> h+16
    EMA_DECAY = 0.999
    WD = 1e-4

    W_B = 0.6
    CALIBRATE = True
    CAL_RANGE = (0.90, 1.10)
    INFER_BATCH = 8
    WORKERS = 2
    MAX_HOURS = 10.5
    SEED = 42


def seed_all(seed=C.SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def art(name):
    p = os.path.join(C.WORK, name)
    if os.path.exists(p):
        return p
    if C.PREV:
        p = os.path.join(C.PREV, name)
        if os.path.exists(p):
            return p
    return None


os.makedirs(C.WORK, exist_ok=True)
seed_all()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', DEVICE)

# %% [markdown]
# ## 1. Official episode masks (robust STL, period=24) — identical to v3

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
    print(f'  computing STL episode masks for {month}...')
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
# ## 2. Stats with physics-derived features + diurnal climatology
#
# Per-pixel mean/std (train region only) for 12 dynamic channels:
# 9 met + wind speed + ventilation coefficient (log1p(pblh*ws)) + advection
# tendency of log PM2.5. Plus: per-pixel 24-hour climatology of log PM2.5
# (for the anomaly channel), per-pixel std of the hourly log PM2.5 tendency,
# and diurnal swdown/t2 templates for test-time hour estimation.

# %%
def month_val_start(T):
    return T - int(T * C.VAL_FRAC)


def _derived(arrs):
    """ws, vent, adv from raw arrays. Works for (T,H,W) with any T."""
    ws = np.sqrt(arrs['u10'] ** 2 + arrs['v10'] ** 2)
    vent = np.log1p(arrs['pblh'] * ws)
    logc = np.log1p(arrs['cpm25'])
    gy = np.gradient(logc, axis=1)   # along H (north-south)
    gx = np.gradient(logc, axis=2)   # along W (east-west)
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
# ## 3. Datasets — leak-free split, window-local tendency + anomaly channels
#
# `dpm` (hourly log-PM tendency) is computed inside the window with the
# first frame set to 0, identically for train and test (a precomputed month
# diff would give train a real previous-hour value the test windows cannot
# have). `clim_anom` is the deviation of each input frame from the per-pixel
# hour-of-day train climatology, in per-pixel std units.

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
# ## 4. Model — multiscale AFNO (35x31 + 18x16), SimVP translator,
# lead-time conditioned decoder with last+mean skip fusion

# %%
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
        # multiscale: spectral mixing at 35x31 before the bottleneck pool
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


# %% [markdown]
# ## 5. Horizon-weighted loss and the exact competition metric

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


# %% [markdown]
# ## 6. Training — EMA ramp, metric checkpointing, CSV log, resume

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


def run_phase(model, name, train_ds, val_ds, loss_w, ep_wt, lr, epochs,
              batch, patience, oversample=False, init_best=-1.0):
    crit = CompositeLoss(loss_w, ep_wt)
    nw = C.WORKERS
    pw = nw > 0
    if oversample:
        d = train_ds.density
        thr = np.quantile(d[d > 0], 0.7) if (d > 0).any() else 0.01
        w = np.where(d >= thr, 3.0, 1.0)
        sampler = WeightedRandomSampler(torch.from_numpy(w / w.sum()),
                                        num_samples=len(train_ds), replacement=True)
        tl = DataLoader(train_ds, batch_size=batch, sampler=sampler,
                        num_workers=nw, pin_memory=True, drop_last=True,
                        persistent_workers=pw)
    else:
        tl = DataLoader(train_ds, batch_size=batch, shuffle=True,
                        num_workers=nw, pin_memory=True, drop_last=True,
                        persistent_workers=pw)
    vl = DataLoader(val_ds, batch_size=batch, shuffle=False,
                    num_workers=nw, pin_memory=True, persistent_workers=pw)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=C.WD)
    steps = len(tl) * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=steps, pct_start=0.05,
        div_factor=20, final_div_factor=200)
    scaler = torch.amp.GradScaler('cuda', enabled=DEVICE.type == 'cuda')
    ema = EMA(model, C.EMA_DECAY)
    best_path = os.path.join(C.WORK, f'{name}_best.pt')
    last_path = os.path.join(C.WORK, f'{name}_last.pt')

    start_ep, best_score, bad = 1, init_best, 0
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
        bad = ck.get('bad', 0)
        print(f'[{name}] resumed at epoch {start_ep}, best={best_score:.4f}')

    for ep in range(start_ep, epochs + 1):
        if time_left() < 600:
            print(f'[{name}] time budget reached, stopping')
            time_stop = True
            break
        t0 = time.time()
        model.train()
        tloss, nb = 0.0, 0
        tr_acc = defaultdict(float)
        for i, (x, t_log, t_raw, last_log, mask) in enumerate(tl):
            if i % 50 == 0 and i > 0 and time_left() < 600:
                print(f'[{name}] time budget reached mid-epoch, stopping')
                time_stop = True
                break
            x, t_log = x.to(DEVICE, non_blocking=True), t_log.to(DEVICE, non_blocking=True)
            t_raw = t_raw.to(DEVICE, non_blocking=True)
            last_log = last_log.to(DEVICE, non_blocking=True)
            mask = mask.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', enabled=DEVICE.type == 'cuda'):
                pred_log = model(x, last_log)
            loss, mets = crit(pred_log.float(), t_log, t_raw, mask)
            if not torch.isfinite(loss):
                print(f'  [{name} E{ep} {i}] non-finite loss, batch skipped')
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update(model)
            tloss += loss.item(); nb += 1
            for k, v in mets.items():
                tr_acc[k] += v
            if i % 100 == 0:
                print(f'  [{name} E{ep} {i}/{len(tl)}] loss={loss.item():.4f} '
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
            best_score = vm['score']
            bad = 0
            torch.save({'model': ema.state_for_eval(model), 'metrics': vm},
                       best_path)
        else:
            bad += 1
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(),
                    'sched': sched.state_dict(), 'ema': ema.shadow,
                    'ema_n': ema.n, 'epoch': ep, 'best_score': best_score,
                    'bad': bad}, last_path)
        tr_m = {k: v / max(nb, 1) for k, v in tr_acc.items()}
        secs = time.time() - t0
        print(f'[{name}] E{ep}/{epochs} ({secs:.0f}s)')
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
        log_epoch(dict(phase=name, epoch=ep, secs=round(secs),
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
            fh.write(f'best={best_score:.4f}\n')
    if os.path.exists(best_path):
        return best_path
    return art(f'{name}_best.pt')


# %% [markdown]
# ## 7. Per-horizon calibration on validation
#
# For each forecast hour, fit one multiplicative factor maximizing
# (NormGlobalSMAPE + NormEpisodeSMAPE)/2 on the holdout. Pearson correlation
# is scale-invariant, so NormEpisodeCorr is unaffected by construction.
# Factors are clipped to [0.90, 1.10]; 16 scalars fitted on ~850 window-hours
# x 17k pixels cannot meaningfully overfit.

# %%
@torch.no_grad()
def collect_val_preds(model_specs, val_ds):
    """Blended val predictions. model_specs: list of (state_dict, weight)."""
    nw = 0
    vl = DataLoader(val_ds, batch_size=C.INFER_BATCH, shuffle=False,
                    num_workers=nw, pin_memory=True)
    N = len(val_ds)
    pred = np.zeros((N, C.T_OUT, C.H, C.W), np.float32)
    tgt = np.zeros((N, C.T_OUT, C.H, C.W), np.float32)
    msk = np.zeros((N, C.T_OUT, C.H, C.W), np.uint8)
    model = AFNOv4().to(DEVICE)
    wsum = sum(w for _, w in model_specs)
    for sd, w in model_specs:
        model.load_state_dict(sd)
        model.eval()
        i = 0
        for x, t_log, t_raw, last_log, mask in vl:
            x, last_log = x.to(DEVICE), last_log.to(DEVICE)
            with torch.autocast('cuda', enabled=DEVICE.type == 'cuda'):
                pl = model(x, last_log)
            pr = F.relu(torch.expm1(pl.float().clamp(max=13.0)))
            bs = x.shape[0]
            pred[i:i + bs] += (w / wsum) * pr.cpu().numpy()
            if tgt[i:i + bs].sum() == 0:
                tgt[i:i + bs] = t_raw.numpy()
                msk[i:i + bs] = mask.numpy().astype(np.uint8)
            i += bs
    del model
    gc.collect(); torch.cuda.empty_cache()
    return pred, tgt, msk


def fit_horizon_scales(pred, tgt, msk):
    lo, hi = C.CAL_RANGE
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
    print('  horizon scales:', np.round(scales, 3).tolist())
    np.save(os.path.join(C.WORK, 'horizon_scales.npy'), scales)
    return scales


# %% [markdown]
# ## 8. Run: masks -> stats -> data -> seed loop (phase A/B) -> calibration
# -> inference. preds.npy is written after every completed stage.

# %%
def predict(model, test_ds):
    model.eval()
    dl = DataLoader(test_ds, batch_size=C.INFER_BATCH, shuffle=False,
                    num_workers=0, pin_memory=True)
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


def save_preds(preds, tag=''):
    preds = preds.astype(np.float32)
    assert preds.shape == (C.N_TEST, C.H, C.W, C.T_OUT), preds.shape
    assert np.isfinite(preds).all()
    np.save(os.path.join(C.WORK, 'preds.npy'), preds)
    print(f'  preds.npy written{" (" + tag + ")" if tag else ""}  '
          f'range=[{preds.min():.2f},{preds.max():.2f}]  mean={preds.mean():.3f}')


def main():
    print('=' * 60, '\nSTEP 1: stats + episode masks + data\n', '=' * 60)
    stats = build_stats()
    months, static = load_months(stats)

    print('=' * 60, '\nSTEP 2: datasets\n', '=' * 60)
    train_ds = PM25Windows(months, static, stats, 'train')
    val_ds = PM25Windows(months, static, stats, 'val')

    n_par = sum(p.numel() for p in AFNOv4().parameters())
    print(f'AFNOv4 params: {n_par/1e6:.2f}M  seeds: {C.SEEDS}')

    test_ds = PM25Test(stats, static)
    seed_specs = []     # per-seed list of (state_dict, weight) for calibration
    seed_preds = []     # per-seed blended test predictions

    for seed in C.SEEDS:
        tag = f's{seed}'
        a_name, b_name = f'{tag}_phaseA', f'{tag}_phaseB'
        a_done = art(f'{a_name}_done.flag') and art(f'{a_name}_best.pt')
        b_done = art(f'{b_name}_done.flag') and art(f'{b_name}_best.pt')
        a_trained = False

        seed_all(seed)
        model = AFNOv4().to(DEVICE)

        if not b_done:
            if not a_done:
                if time_left() < 1200:
                    print(f'[{tag}] no time for phase A, skipping seed')
                    del model
                    continue
                print('=' * 60, f'\n{tag} phase A (global training)\n', '=' * 60)
                run_phase(model, a_name, train_ds, val_ds,
                          C.LOSS_A, C.EP_WT_A, C.A_LR, C.A_EPOCHS,
                          C.A_BATCH, C.A_PATIENCE)
                a_trained = True
            a_best = art(f'{a_name}_best.pt')
            if a_best:
                ck = torch.load(a_best, map_location=DEVICE, weights_only=False)
                model.load_state_dict(ck['model'])
                a_score = ck.get('metrics', {}).get('score', -1.0)
                print(f'{tag} phase A best:', ck.get('metrics'))
                if time_left() > 1800:
                    print('=' * 60, f'\n{tag} phase B (episode fine-tune)\n', '=' * 60)
                    run_phase(model, b_name, train_ds, val_ds,
                              C.LOSS_B, C.EP_WT_B, C.B_LR, C.B_EPOCHS,
                              C.B_BATCH, C.B_PATIENCE, oversample=True,
                              init_best=a_score)
            else:
                print(f'WARNING: {tag} has no phase A checkpoint')

        # per-seed inference (also serves as the safety preds.npy)
        a_best = art(f'{a_name}_best.pt')
        b_local = os.path.join(C.WORK, f'{b_name}_best.pt')
        if os.path.exists(b_local):
            b_best = b_local
        else:
            b_best = None if a_trained else art(f'{b_name}_best.pt')

        sp, wsum = None, 0.0
        for path, w in [(b_best, C.W_B), (a_best, 1 - C.W_B)]:
            if not path:
                continue
            ck = torch.load(path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ck['model'])
            p = predict(model, test_ds)
            if not np.isfinite(p).all():
                print(f'  WARNING: non-finite preds from {os.path.basename(path)}')
                continue
            print(f'  {os.path.basename(path)}: '
                  f'range=[{p.min():.1f},{p.max():.1f}] '
                  f'(val {ck.get("metrics", {}).get("score", 0):.4f})')
            seed_specs.append((ck['model'], w))
            sp = p * w if sp is None else sp + p * w
            wsum += w
        del model
        gc.collect(); torch.cuda.empty_cache()
        if sp is not None:
            sp /= wsum
            seed_preds.append(sp)
            np.save(os.path.join(C.WORK, f'preds_{tag}.npy'), sp)
            save_preds(np.mean(seed_preds, axis=0), tag=f'after {tag}')

    print('=' * 60, '\nSTEP 9: calibration + final preds\n', '=' * 60)
    if not seed_preds:
        print('WARNING: no usable model - writing persistence fallback')
        last = np.asarray(test_ds.arrs['cpm25'][:, -1], np.float32)
        save_preds(np.repeat(last[..., None], C.T_OUT, axis=-1), tag='persistence')
        return

    preds = np.mean(seed_preds, axis=0)

    if C.CALIBRATE and time_left() > 300:
        print('  fitting per-horizon calibration on validation...')
        vp, vt, vm = collect_val_preds(seed_specs, val_ds)
        scales = fit_horizon_scales(vp, vt, vm)
        preds = preds * scales[None, None, None, :]
        del vp, vt, vm

    del months, train_ds, val_ds
    gc.collect()
    save_preds(preds, tag='final')


if __name__ == '__main__':
    main()
