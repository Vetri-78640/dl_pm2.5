# %% [markdown]
# ## 15. ML + DL hybrid: a gradient-boosted correction of the ResGRU-UNet ensemble
#
# A LightGBM model predicts the error the network ensemble is about to make, and the forecast
# is corrected by it. No network is trained here.
#
# - Base forecast: the uncalibrated ResGRU-UNet 5-seed mean, by inference from the study checkpoints.
# - Target: log1p(truth) - log1p(base forecast), per pixel and lead hour.
# - Features (33): the 20 input channels at the last input hour, log PM2.5 at the 9 earlier
#   hours (the ridge features without the bias), the base forecast (log), its change from the
#   last hour, the spread of the 5 seeds (log), and the lead hour.
# - One model for all lead hours, fitted on every 4th pixel in each direction.
#
# Rule, written on 2026-11-10 before the first run. Settings below are fixed; nothing is tuned
# after seeing a result.
# 1. Leave one month out within validation: fit on the val windows of three months, predict the
#    fourth, for all four months. Every reported hybrid number is out of fold.
# 2. Compare with the uncalibrated 5-seed ensemble on the same windows: benchmark score, MAE,
#    RMSE and R², pooled and per held-out month.
# 3. The hybrid is "better" only if its pooled benchmark score is higher AND its score is higher
#    in at least 3 of the 4 held-out months. Otherwise the verdict is "no gain".
# 4. Only if the verdict is "better": fit on all four months and write test predictions.
# Limit: the val windows also picked each network's best epoch, so this tests the correction,
# not the whole system on fully unseen data.

# %%
HYB_STRIDE = 4
HYB_PARAMS = dict(objective='huber', alpha=0.5, n_estimators=300, learning_rate=0.05, num_leaves=31,
                  min_child_samples=500, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                  reg_lambda=10.0, random_state=0, n_jobs=-1, verbose=-1)
HYB_FEATS = ([f'{f}_last' for f in C.FEATS] + ['wind_speed', 'ventilation', 'advection', 'tendency', 'hour_sin', 'hour_cos',
             'diurnal_anomaly', 'lat', 'lon', 'pm_climatology', 'stl_sigma']
             + [f'pm_h-{k}' for k in range(C.T_IN - 1, 0, -1)] + ['base_log', 'base_change', 'seed_spread', 'lead_hour'])
HYB_ENSEMBLE_REF = 0.9193


def hybrid_sel():
    sel = np.zeros((C.H, C.W), bool)
    sel[::HYB_STRIDE, ::HYB_STRIDE] = True
    return sel.ravel()


def hybrid_rows(x, last_log, base_log, spread, sel=None):
    """One window -> (T_OUT * pixels, 33) float32, lead hour major."""
    f = ridge_features(x)[:, :-1].astype(np.float32)
    ll = np.asarray(last_log, np.float32).ravel()
    b = base_log.reshape(C.T_OUT, -1)
    sp = spread.reshape(C.T_OUT, -1)
    if sel is not None:
        f, ll, b, sp = f[sel], ll[sel], b[:, sel], sp[:, sel]
    k = f.shape[1]
    out = np.empty((C.T_OUT, len(f), k + 4), np.float32)
    out[:, :, :k] = f[None]
    out[:, :, k] = b
    out[:, :, k + 1] = b - ll[None]
    out[:, :, k + 2] = sp
    out[:, :, k + 3] = np.arange(1, C.T_OUT + 1, dtype=np.float32)[:, None]
    return out.reshape(-1, k + 4)


def resgru_ensemble(ds, test=False):
    """5-seed mean forecast as log1p, the seed spread of log1p forecasts, and (val) truth and mask."""
    s1 = sl = sl2 = tgt = msk = None
    for s in SEEDS5:
        model = BACKBONES['resgru']().to(DEVICE)
        model.load_state_dict(torch.load(ckpt_of('resgru', s), map_location=DEVICE, weights_only=False)['model'])
        if test:
            p = predict(model, ds, AB).transpose(0, 3, 1, 2)
        else:
            p, tgt, msk = val_arrays(model, ds)
        lp = np.log1p(p.astype(np.float64))
        s1 = p.astype(np.float64) if s1 is None else s1 + p
        sl = lp if sl is None else sl + lp
        sl2 = lp * lp if sl2 is None else sl2 + lp * lp
        del model, p, lp
        gc.collect()
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
    n = len(SEEDS5)
    base = (s1 / n).astype(np.float32)
    spread = np.sqrt(np.maximum(sl2 / n - (sl / n) ** 2, 0)).astype(np.float32)
    return base, np.log1p(base), spread, tgt, msk


def hybrid_fit(val_ds, idx, base_log, spread, tgt):
    import lightgbm as lgb
    sel = hybrid_sel()
    X, y = [], []
    for i in idx:
        x, _, _, last_log, _ = val_ds[i]
        X.append(hybrid_rows(x, last_log, base_log[i], spread[i], sel))
        y.append((np.log1p(tgt[i]) - base_log[i]).reshape(C.T_OUT, -1)[:, sel].ravel())
    X, y = np.concatenate(X), np.concatenate(y)
    return lgb.LGBMRegressor(**HYB_PARAMS).fit(X, y), len(y)


def hybrid_apply(model, x, last_log, base_log, spread):
    """Corrected forecast (T_OUT, H, W) in ug/m3. model None: zero correction."""
    c = 0.0 if model is None else model.predict(hybrid_rows(x, last_log, base_log, spread)).reshape(C.T_OUT, C.H, C.W)
    return np.maximum(np.expm1(np.minimum(base_log.astype(np.float64) + c, 13.0)), 0.0).astype(np.float32)


def hybrid_scores(pred, tgt, msk):
    m = reg_metrics(pred, tgt, msk, truth_stats(tgt, msk))
    m.update(terms_of(wh_terms(pred, tgt, msk)))
    return {k: m[k] for k in ('score', 'gsmape', 'esmape', 'ecorr', 'mae', 'rmse', 'r2')}


def hybrid():
    t0 = time.time()
    if C.IS_KAGGLE:
        check_gpu()
    miss = [s for s in SEEDS5 if not ckpt_of('resgru', s)]
    if miss:
        raise RuntimeError(f'missing ResGRU-UNet checkpoints for seeds {miss}')
    stats = build_stats()
    months, static = load_months(stats)
    val_ds = PM25Windows(months, static, stats, 'val')
    keys = list(val_ds.keys)
    mon = np.array([mi for mi, _ in val_ds.samples])
    base, base_log, spread, tgt, msk = resgru_ensemble(val_ds)
    zero = np.stack([hybrid_apply(None, None, None, base_log[i], None) for i in range(len(val_ds))])
    oof = np.zeros_like(base)
    folds, seen = {}, []
    for mi, k in enumerate(keys):
        tr, te = np.nonzero(mon != mi)[0], np.nonzero(mon == mi)[0]
        seen.append(not set(mon[tr]) & {mi} and len(set(tr) & set(te)) == 0)
        model, n = hybrid_fit(val_ds, tr, base_log, spread, tgt)
        for i in te:
            x, _, _, last_log, _ = val_ds[i]
            oof[i] = hybrid_apply(model, x, last_log, base_log[i], spread[i])
        folds[k] = dict(n_fit_rows=n, n_fit_windows=len(tr), n_test_windows=len(te),
                        ensemble=hybrid_scores(base[te], tgt[te], msk[te]), hybrid=hybrid_scores(oof[te], tgt[te], msk[te]))
        folds[k]['score_gain'] = round(folds[k]['hybrid']['score'] - folds[k]['ensemble']['score'], 5)
        print(f"  held out {k}: ensemble {folds[k]['ensemble']['score']:.5f}  hybrid {folds[k]['hybrid']['score']:.5f}  "
              f"gain {folds[k]['score_gain']:+.5f}  ({time.time() - t0:.0f} s)")
        del model
        gc.collect()
    pe, ph, pz = hybrid_scores(base, tgt, msk), hybrid_scores(oof, tgt, msk), hybrid_scores(zero, tgt, msk)
    wins = sum(f['score_gain'] > 0 for f in folds.values())
    better = ph['score'] > pe['score'] and wins >= 3
    ck = {'zero correction reproduces the ensemble score (within 1e-5)': abs(pz['score'] - pe['score']) < 1e-5,
          'no fold is fitted on its own month': all(seen),
          'every val window is predicted out of fold exactly once': bool((np.bincount(mon, minlength=len(keys)) > 0).all()
                                                                         and sum(f['n_test_windows'] for f in folds.values()) == len(val_ds)),
          'hybrid predictions finite and non-negative': bool(np.isfinite(oof).all() and oof.min() >= 0)}
    if C.IS_KAGGLE:
        ck['ensemble score equals the 5-seed analysis (0.9193 within 1e-4)'] = abs(pe['score'] - HYB_ENSEMBLE_REF) < 1e-4
    res = dict(rule='better only if the pooled out-of-fold score is higher and the hybrid wins in at least 3 of 4 held-out months',
               rule_date='2026-11-10', verdict='better' if better else 'no gain', months_won=int(wins),
               pooled=dict(ensemble=pe, hybrid=ph, score_gain=round(ph['score'] - pe['score'], 5)), folds=folds,
               mean_abs_correction_log=round(float(np.abs(np.log1p(oof) - base_log).mean()), 5),
               params=HYB_PARAMS, features=HYB_FEATS, pixel_stride=HYB_STRIDE, n_val_windows=len(val_ds),
               limit='the val windows also picked the best epoch of each network; this tests the correction only')
    del months
    gc.collect()
    model, n = hybrid_fit(val_ds, np.arange(len(val_ds)), base_log, spread, tgt)
    gain = model.booster_.feature_importance('gain')
    res['importance_gain_share_all_months_fit'] = {k: round(float(v), 4) for k, v in
                                                   sorted(zip(HYB_FEATS, gain / gain.sum()), key=lambda t: -t[1])}
    if better:
        test_ds = PM25Test(stats, static)
        _, tl, tsp, _, _ = resgru_ensemble(test_ds, test=True)
        tp = np.stack([hybrid_apply(model, *test_ds[i], tl[i], tsp[i]) for i in range(len(test_ds))]).transpose(0, 2, 3, 1)
        ck['test predictions (218, 140, 124, 16), finite, non-negative'] = bool(
            tp.shape == (C.N_TEST, C.H, C.W, C.T_OUT) and np.isfinite(tp).all() and tp.min() >= 0)
        np.save(os.path.join(C.WORK, 'preds_hybrid.npy'), tp.astype(np.float32))
    res.update(checks=ck, all_checks_pass=all(ck.values()), minutes=round((time.time() - t0) / 60, 1))
    json.dump(res, open(os.path.join(C.WORK, 'hybrid.json'), 'w'), indent=1)
    print(f"  pooled: ensemble {pe['score']:.5f}  hybrid {ph['score']:.5f}  months won {wins}/4  verdict: {res['verdict']}")
    print('  checks:', 'all pass' if res['all_checks_pass'] else ck)


if __name__ == '__main__':
    hybrid()
