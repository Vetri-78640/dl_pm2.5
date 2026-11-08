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


if __name__ == '__main__':
    baselines()
