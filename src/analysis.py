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


if __name__ == '__main__':
    analyze()
