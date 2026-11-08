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


if __name__ == '__main__':
    regression()
