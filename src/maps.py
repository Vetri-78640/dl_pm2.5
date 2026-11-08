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
