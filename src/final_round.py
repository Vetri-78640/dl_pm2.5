# %% [markdown]
# ## 9. Final round: 2 seeds, ResGRU-UNet vs AFNO V4
#
# Runs after seed 1337 has trained. Both families get V4's full recipe as it was
# actually submitted: the equal-weight mean of the seed 42 and seed 1337 phase A
# best checkpoints, then per-horizon calibration fitted on validation (V4's
# phase B never beat phase A, so it never entered V4's blend).
#
# Inputs: ResGRU-UNet seed 42 from the round-one study output, seed 1337 from
# this run; AFNO V4 seeds 42 and 1337 from the V4 output. Every model is scored
# here on the same 214 validation windows with the competition metric. Phase B
# is not run for ResGRU-UNet: it did not enter V4's final, and skipping it can
# only hold ResGRU-UNet back, so the comparison is conservative for the challenger.
#
# **Decision rule (fixed before this run).** Val gap = ResGRU-UNet minus AFNO V4,
# both 2-seed means without calibration, scored here (out of sample for both).
# Private = full recipe (2 seeds + calibration); AFNO V4 is 0.8833.
# - ResGRU-UNet better: val gap > +0.003 and private above 0.8833
# - AFNO V4 better: val gap < -0.003 and private below 0.8833
# - tie: |val gap| <= 0.003
# - inconclusive: val and private disagree
#
# Secondary, reported but not part of the verdict: 2-seed private without
# calibration (AFNO V4 0.8779) and the calibrated val gap (in sample, since the
# scales are fitted on val).
#
# Checks (reported in `summary.json`, the run does not stop on them):
# - each checkpoint's val score here equals the score logged when it was trained
# - AFNO V4 test predictions here equal V4's own `preds_s42.npy` / `preds_s1337.npy`
# - refitting V4's calibration here gives V4's own `horizon_scales.npy`
#
# Outputs: `preds.npy` (ResGRU-UNet 2 seeds + calibration), `preds_2seed.npy`
# (ResGRU-UNet 2 seeds, no calibration), `horizon_scales_resgru.npy`,
# `horizon_scales_afnov4_refit.npy`; results under `final_round` in `summary.json`.

# %%
FR_RESGRU_DIR = (find_dir('/kaggle/input', ['resgru_s42_phaseA_best.pt'])
                 if C.IS_KAGGLE else None)
FR_BATCH = 4
CAL_RANGE = (0.90, 1.10)
v4 = lambda f: os.path.join(C.V4_DIR, f) if C.V4_DIR else None


def final_inputs():
    return {'ResGRU-UNet seed 42 (study output)':
                os.path.join(FR_RESGRU_DIR, 'resgru_s42_phaseA_best.pt') if FR_RESGRU_DIR else None,
            'AFNO V4 seed 42': v4('s42_phaseA_best.pt'),
            'AFNO V4 seed 1337': v4('s1337_phaseA_best.pt'),
            'AFNO V4 preds_s42.npy': v4('preds_s42.npy'),
            'AFNO V4 preds_s1337.npy': v4('preds_s1337.npy'),
            'AFNO V4 horizon_scales.npy': v4('horizon_scales.npy')}


def final_precheck():
    """Stop before any training if an input of the final round is missing."""
    print('=' * 60, '\nFINAL ROUND PRE-CHECK: inputs\n', '=' * 60)
    missing = []
    for k, p in final_inputs().items():
        ok = bool(p) and os.path.exists(p)
        print(f'  {"ok     " if ok else "MISSING"} {k}: {p}')
        if not ok:
            missing.append(k)
    if missing:
        raise RuntimeError(f'final round inputs missing: {missing}')


@torch.no_grad()
def val_arrays(model, val_ds):
    model.eval()
    vl = DataLoader(val_ds, batch_size=FR_BATCH, shuffle=False, num_workers=C.WORKERS, pin_memory=True)
    N = len(val_ds)
    pred = np.zeros((N, C.T_OUT, C.H, C.W), np.float32)
    tgt = np.zeros((N, C.T_OUT, C.H, C.W), np.float32)
    msk = np.zeros((N, C.T_OUT, C.H, C.W), np.uint8)
    i = 0
    for x, t_log, t_raw, last_log, mask in vl:
        x, last_log = x.to(DEVICE), last_log.to(DEVICE)
        with torch.autocast('cuda', enabled=DEVICE.type == 'cuda'):
            pl = model(x, last_log)
        pr = F.relu(torch.expm1(pl.float().clamp(max=13.0)))
        bs = x.shape[0]
        pred[i:i + bs] = pr.cpu().numpy()
        tgt[i:i + bs] = t_raw.numpy()
        msk[i:i + bs] = mask.numpy().astype(np.uint8)
        i += bs
    return pred, tgt, msk


@torch.no_grad()
def score_arrays(pred, tgt, msk):
    """Competition metric on arrays, through the same comp_metric_* code as evaluate()."""
    acc = defaultdict(list)
    for i in range(0, len(pred), FR_BATCH):
        p = torch.from_numpy(pred[i:i + FR_BATCH]).to(DEVICE)
        t = torch.from_numpy(tgt[i:i + FR_BATCH]).to(DEVICE)
        m = torch.from_numpy(msk[i:i + FR_BATCH].astype(np.float32)).to(DEVICE)
        comp_metric_batch(p, t, m, acc)
    return comp_metric_final(acc)


def fit_horizon_scales(pred, tgt, msk, tag):
    """V4's calibration, verbatim except the output file name."""
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
    print(f'  {tag} horizon scales:', np.round(scales, 3).tolist())
    np.save(os.path.join(C.WORK, f'horizon_scales_{tag}.npy'), scales)
    return scales


def terms(m):
    return {k: round(m[k], 5) for k in ('score', 'gsmape', 'esmape', 'ecorr')}


def final_round():
    print('=' * 60, '\nSTEP 7: final round - 2 seeds, ResGRU-UNet vs AFNO V4\n', '=' * 60)
    sp = os.path.join(C.WORK, 'summary.json')
    summary = json.load(open(sp))
    if summary.get('status') != 'complete':
        print('  seed 1337 did not complete; final round skipped')
        return
    final_precheck()
    stats = build_stats()
    months, static = load_months(stats)
    val_ds = PM25Windows(months, static, stats, 'val')
    test_ds = PM25Test(stats, static)
    inp = final_inputs()
    fams = {'resgru': (lambda: ResGRUUNet(C.C_IN),
                       [inp['ResGRU-UNet seed 42 (study output)'],
                        os.path.join(C.WORK, 'resgru_s1337_phaseA_best.pt')]),
            'afnov4': (AFNOv4, [inp['AFNO V4 seed 42'], inp['AFNO V4 seed 1337']])}
    res, test_mean, tgt, msk = {}, {}, None, None
    for fam, (make, paths) in fams.items():
        vps, tps, seeds = [], [], {}
        for seed, path in zip((42, 1337), paths):
            ck = torch.load(path, map_location=DEVICE, weights_only=False)
            model = make().to(DEVICE)
            model.load_state_dict(ck['model'])
            vp, t, m = val_arrays(model, val_ds)
            if tgt is None:
                tgt, msk = t, m
            tp = predict(model, test_ds, FR_BATCH)
            del model
            gc.collect(); torch.cuda.empty_cache()
            s = score_arrays(vp, tgt, msk)
            logged = float(ck.get('metrics', {}).get('score', float('nan')))
            seeds[seed] = dict(**terms(s), logged=round(logged, 5), epoch=ck.get('epoch'))
            print(f'  {fam} seed {seed}: val {s["score"]:.5f} (logged {logged:.5f})')
            vps.append(vp); tps.append(tp)
        vmean = np.mean(vps, axis=0)
        test_mean[fam] = np.mean(tps, axis=0)
        s2 = score_arrays(vmean, tgt, msk)
        scales = fit_horizon_scales(vmean, tgt, msk, 'resgru' if fam == 'resgru' else 'afnov4_refit')
        s2c = score_arrays(vmean * scales[None, :, None, None], tgt, msk)
        print(f'  {fam} 2 seeds: val {s2["score"]:.5f}; with calibration (fitted on val) {s2c["score"]:.5f}')
        res[fam] = dict(seeds=seeds, two_seed=terms(s2), two_seed_cal_in_sample=terms(s2c),
                        scales=[round(float(v), 3) for v in scales],
                        per_seed_test=tps if fam == 'afnov4' else None)
        del vps, vmean
        gc.collect()

    checks = {}
    for fam in res:
        for seed, d in res[fam]['seeds'].items():
            checks[f'{fam} s{seed} val equals logged'] = abs(d['score'] - d['logged']) < 1e-4
    for seed, tp in zip((42, 1337), res['afnov4'].pop('per_seed_test')):
        ref = np.load(inp[f'AFNO V4 preds_s{seed}.npy'])
        rel = float(np.abs(tp - ref).mean() / np.abs(ref).mean())
        print(f'  AFNO V4 seed {seed} test preds vs V4 output: mean relative diff {rel:.2e}')
        checks[f'afnov4 s{seed} test preds equal V4 output (rel < 1e-3)'] = rel < 1e-3
    v4_scales = np.load(inp['AFNO V4 horizon_scales.npy'])
    n_eq = int(np.isclose(v4_scales, res['afnov4']['scales'], atol=1e-3).sum())
    print(f'  refitted AFNO V4 scales equal V4 horizon_scales.npy in {n_eq}/16 hours')
    checks['afnov4 refitted scales equal V4 output (16/16)'] = n_eq == C.T_OUT
    res['resgru'].pop('per_seed_test')

    preds_2 = test_mean['resgru']
    np.save(os.path.join(C.WORK, 'preds_2seed.npy'), preds_2.astype(np.float32))
    scales_r = np.array(res['resgru']['scales'], np.float32)
    save_preds(preds_2 * scales_r[None, None, None, :])

    gap = res['resgru']['two_seed']['score'] - res['afnov4']['two_seed']['score']
    summary['final_round'] = dict(res, val_gap_two_seed=round(gap, 5),
                                  val_gap_two_seed_cal_in_sample=round(
                                      res['resgru']['two_seed_cal_in_sample']['score']
                                      - res['afnov4']['two_seed_cal_in_sample']['score'], 5),
                                  checks=checks, all_checks_pass=all(checks.values()))
    json.dump(summary, open(sp, 'w'), indent=1)
    print(f'  val gap, 2 seeds (ResGRU-UNet - AFNO V4): {gap:+.5f} '
          f'({"beyond" if abs(gap) > 0.003 else "within"} the 0.003 tie margin)')
    print('  checks:', 'all pass' if all(checks.values()) else checks)
    print('  final round written to summary.json')


# %%
if __name__ == '__main__':
    final_precheck()
    main()
    final_round()
