# %% [markdown]
# ## 13. Ridge regression baseline
#
# A linear model per pixel, shared over the grid. Features: the 20 normalized input channels
# at the last input hour, plus the normalized log PM2.5 channel at the 9 earlier input hours,
# plus a bias (30 values). Targets: the 16 changes log1p(y_t) - log1p(y_last), the same
# residual form the networks use. Closed-form ridge on train windows (every 4th pixel in each
# direction), penalty chosen on the val score from a fixed grid. Scored like section 11.
# CPU only.

# %%
RIDGE_LAMBDAS = [1e-2, 1e0, 1e2, 1e4, 1e6]
RIDGE_STRIDE = 4


def ridge_features(x):
    """x: (20, T_IN, H, W) tensor or array -> (H*W, 30) float64."""
    x = np.asarray(x, np.float64)
    f = np.concatenate([x[:, -1], x[0, :-1], np.ones((1, C.H, C.W))], 0)
    return f.reshape(f.shape[0], -1).T


def ridge_fit(train_ds):
    k = C.C_IN + C.T_IN - 1 + 1
    xtx, xty, n = np.zeros((k, k)), np.zeros((k, C.T_OUT)), 0
    sel = np.zeros((C.H, C.W), bool)
    sel[::RIDGE_STRIDE, ::RIDGE_STRIDE] = True
    sel = sel.ravel()
    for i in range(len(train_ds)):
        x, t_log, t_raw, last_log, mask = train_ds[i]
        f = ridge_features(x)[sel]
        y = (t_log.numpy().astype(np.float64) - last_log.numpy()[None]).reshape(C.T_OUT, -1).T[sel]
        xtx += f.T @ f
        xty += f.T @ y
        n += len(f)
    return xtx, xty, n


def ridge_weights(xtx, xty, lam):
    reg = lam * np.eye(len(xtx))
    reg[-1, -1] = 0.0
    return np.linalg.solve(xtx + reg, xty)


def ridge_predict(x, last_log, w):
    d = (ridge_features(x) @ w).T.reshape(C.T_OUT, C.H, C.W)
    pl = np.asarray(last_log, np.float64)[None] + d
    return np.maximum(np.expm1(np.minimum(pl, 13.0)), 0.0).astype(np.float32)


def ridge():
    t0 = time.time()
    stats = build_stats()
    months, static = load_months(stats)
    train_ds = PM25Windows(months, static, stats, 'train')
    val_ds = PM25Windows(months, static, stats, 'val')
    xtx, xty, n = ridge_fit(train_ds)
    print(f'  fitted on {n:,} pixel-windows ({len(train_ds)} train windows)')
    tgt, msk, per = val_truth(val_ds)
    ts = truth_stats(tgt, msk)
    items = [val_ds[i] for i in range(len(val_ds))]
    grid, preds = {}, {}
    for lam in RIDGE_LAMBDAS + ['zero']:
        w = np.zeros((xtx.shape[0], C.T_OUT)) if lam == 'zero' else ridge_weights(xtx, xty, lam)
        p = np.stack([ridge_predict(it[0], it[3], w) for it in items])
        grid[str(lam)] = terms_of(wh_terms(p, tgt, msk))['score']
        if lam != 'zero':
            preds[lam] = (p, w)
        print(f'  lambda {lam}: val score {grid[str(lam)]:.5f}')
    best = max(RIDGE_LAMBDAS, key=lambda l: grid[str(l)])
    p, w = preds[best]
    m = reg_metrics(p, tgt, msk, ts)
    m.update(terms_of(wh_terms(p, tgt, msk)))
    m.update(label=f'ridge regression (lambda {best:g})', lam=best)
    ck = {'zero weights reproduce persistence (score within 1e-4)': abs(grid['zero'] - terms_of(wh_terms(per, tgt, msk))['score']) < 1e-4,
          'val predictions finite and non-negative': bool(np.isfinite(p).all() and p.min() >= 0)}
    del months, items, preds
    gc.collect()
    test_ds = PM25Test(stats, static)
    tp = np.stack([ridge_predict(*test_ds[i], w) for i in range(len(test_ds))]).transpose(0, 2, 3, 1).astype(np.float32)
    ck['test predictions (218, 140, 124, 16), finite, non-negative'] = bool(
        tp.shape == (C.N_TEST, C.H, C.W, C.T_OUT) and np.isfinite(tp).all() and tp.min() >= 0)
    np.save(os.path.join(C.WORK, 'preds_ridge.npy'), tp)
    res = dict(model=m, lambda_grid=grid, n_train_rows=n, features='20 channels at the last hour + log PM2.5 at 9 earlier hours + bias',
               weights=[[round(float(v), 6) for v in row] for row in w], checks=ck, all_checks_pass=all(ck.values()),
               minutes=round((time.time() - t0) / 60, 1))
    json.dump(res, open(os.path.join(C.WORK, 'ridge.json'), 'w'), indent=1)
    print(f'  ridge: MAE {m["mae"]:.3f} RMSE {m["rmse"]:.3f} R2 {m["r2"]:.4f} score {m["score"]:.5f}; checks:',
          'all pass' if res['all_checks_pass'] else ck)


if __name__ == '__main__':
    ridge()
