"""Apply the pre-registered ablation rule (pm25-forecasting 09, ablations) to the run outputs.

usage: python3 ablation_verdict.py <analysis.json> <dir with summary_<variant>.json> [<out.json>]

Reference per family: mean m and sd s of the 5 single-seed val scores (analysis.json).
t = max(2s, 0.002); d = variant val - m; G = m(resgru) - m(afnov4).
supported: d beyond t in the predicted direction and the named term moves the same way
contradicted: d beyond t in the opposite direction; not resolved: |d| <= t
"""
import json, os, sys

# variant: (family, predicted sign of d, term that should move, sign of the term change if it helps the score)
RULES = {
    'afnov4_allskip': ('afnov4', +1, 'gsmape', -1),
    'afnov4_pad': ('afnov4', +1, None, 0),
    'afnov4_nosparse': ('afnov4', +1, 'gsmape', -1),
    'resgru_norec': ('resgru', -1, 'ecorr', +1),
}

an = json.load(open(sys.argv[1]))
ref = an['single_seed_val']
seeds = an['seeds_val']
G = ref['resgru']['mean'] - ref['afnov4']['mean']
out = {'gap_G': round(G, 5), 'variants': {}}
print(f'G (ResGRU-UNet minus AFNO V4, 5-seed means) = {G:+.5f}')
for v, (fam, sign, term, tsign) in RULES.items():
    p = os.path.join(sys.argv[2], f'summary_{v}.json')
    if not os.path.exists(p):
        print(f'{v}: summary missing')
        continue
    sm = json.load(open(p))
    if sm.get('status') != 'complete':
        print(f'{v}: status {sm.get("status")}')
        continue
    b = sm['best']
    m, s = ref[fam]['mean'], ref[fam]['sd']
    t = max(2 * s, 0.002)
    d = b['score'] - m
    tm = {k: sum(seeds[fam][str(x)][k] for x in seeds[fam]) / len(seeds[fam]) for k in ('gsmape', 'esmape', 'ecorr')}
    dt = {k: round(b[k] - tm[k], 5) for k in tm}
    term_ok = term is None or dt[term] * sign * tsign > 0
    if abs(d) <= t:
        verdict = 'not resolved'
    elif d * sign > 0:
        verdict = 'supported' if term_ok else 'score moved as predicted, term did not'
    else:
        verdict = 'contradicted'
    out['variants'][v] = dict(family=fam, val=b['score'], m=m, s=s, t=round(t, 5), d=round(d, 5),
                              share_of_G=round(d / G, 3) if G else None, term_change=dt,
                              predicted_term=term, verdict=verdict, epoch=b.get('epoch'))
    print(f'{v}: val {b["score"]:.5f}  d {d:+.5f}  t {t:.5f}  d/G {d / G:+.2f}  terms {dt}  -> {verdict}')
if len(sys.argv) > 3:
    json.dump(out, open(sys.argv[3], 'w'), indent=1)
