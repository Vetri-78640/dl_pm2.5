"""Build the 5-seed extension notebooks: ResGRU-UNet and AFNO V4, seeds 1, 2, 3,
same code and recipe as the final round (phase A only, no post-processing).
Output: kaggle/seeds_<backbone>/<slug>.ipynb and kernel-metadata.json.
"""
import json, os, re
import build_notebooks as B

SEEDS = [1, 2, 3]
FAM = {
    'resgru': dict(title='ResGRU-UNet', short='ResGRU', arch=B.MODELS['resgru']['arch']),
    'afnov4': dict(title='AFNO V4', short='AFNO V4',
                   arch=('The AFNO V4 network (`AFNOv4`, the class used in every pipeline check): '
                         'per-hour U-Net encoder, 1 AFNO block at 35x31, 6 at 18x16, a SimVP-style '
                         'temporal translator 10 -> 16 hours, 2 AFNO blocks per lead hour, U-Net '
                         'decoder with [last hour, mean] skips. 6.98M parameters.')),
}
HEADER = [
    ('# | seed | 42, one run |',
     '# | seeds | 1, 2, 3 in this notebook (42 and 1337 come from earlier runs) |'),
    ('# **Reference (not retrained):**', '# **Round-one reference (not retrained):**'),
    ('# **Decision rule (fixed before training):**', '# **Round-one decision rule:**'),
]
LOOP = f"""

# %% [markdown]
# ## 9. Seed loop
#
# Seeds {', '.join(map(str, SEEDS))}, one after another, each with the full phase A of
# section 6. After each seed, `summary.json` and `preds.npy` are renamed with the seed
# (`summary_s<seed>.json`, `preds_s<seed>.npy`); checkpoints are already named by seed.
# If the session stops, attach this output as an input and run again: finished seeds
# are skipped and an interrupted seed resumes from its last epoch.

# %%
SEEDS = {SEEDS}


def run_seeds():
    for s in SEEDS:
        C.SEED = s
        if C.IS_KAGGLE:
            C.RESUME_DIR = find_dir('/kaggle/input', [f'{{BACKBONE}}_s{{s}}_phaseA_last.pt'])
        main()
        for f in ('summary.json', 'preds.npy'):
            src = os.path.join(C.WORK, f)
            if os.path.exists(src):
                base, ext = os.path.splitext(f)
                os.replace(src, os.path.join(C.WORK, f'{{base}}_s{{s}}{{ext}}'))
        print(f'seed {{s}} finished')


if __name__ == '__main__':
    run_seeds()
"""


def source(bb):
    s = open(B.SRC).read()
    for a, b in HEADER:
        assert s.count(a) == 1, a
        s = s.replace(a, b)
    s = re.sub(r"^BACKBONE = '[a-z0-9]+'", f"BACKBONE = '{bb}'", s, count=1, flags=re.M)
    s = re.sub(r'^SEED = 42 ', f'SEED = {SEEDS[0]} ', s, count=1, flags=re.M)
    tail = "\n\nif __name__ == '__main__':\n    main()\n"
    assert s.endswith(tail)
    return s[:-len(tail)] + LOOP


def build(bb):
    f = FAM[bb]
    title = f'Code4CleanAir Backbone Study - Seeds 1-3 {f["short"]}'
    assert len(title) <= 50, title
    summary = (f'Seeds 1, 2 and 3 of **{f["title"]}**, trained with exactly the code and recipe of '
               'the final round (phase A only, no post-processing). With seeds 42 and 1337 from '
               'earlier runs, each finalist then has 5 seeds for the uncertainty analysis.')
    notes = f'**{f["title"]} in this notebook**\n# \n# {f["arch"]}'
    src = source(bb)
    out = []
    for kind, text in B.cells_of(src):
        if kind == 'code' and text.startswith('# BACKBONE-CELL:'):
            first, _, rest = text.partition('\n')
            if bb not in first.split(':', 1)[1].split():
                continue
            text = rest
        if kind == 'markdown':
            text = (text.replace('{{TITLE}}', f'seeds 1-3, {f["title"]}').replace('{{SUMMARY}}', summary)
                    .replace('{{ARCH}}', f['arch']).replace('{{MODEL_NOTES}}', notes))
        out.append(B.to_nb_cell(kind, text))
    code = '\n'.join(''.join(c['source']) for c in out if c['cell_type'] == 'code')
    assert f"BACKBONE = '{bb}'" in code and '{{' not in json.dumps(out)
    compile(code, bb, 'exec')
    nb = {'cells': out,
          'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                                      'name': 'python3'},
                       'language_info': {'name': 'python', 'version': '3.10'}},
          'nbformat': 4, 'nbformat_minor': 4}
    d = os.path.join(B.HERE, '..', 'notebooks', 'kaggle', f'seeds_{bb}')
    os.makedirs(d, exist_ok=True)
    fn = f'{B.slug(title)}.ipynb'
    json.dump(nb, open(os.path.join(d, fn), 'w'), indent=1)
    meta = {'id': f'{B.USER}/{B.slug(title)}', 'title': title, 'code_file': fn,
            'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
            'enable_gpu': True, 'enable_tpu': False, 'enable_internet': False,
            'machine_shape': 'NvidiaTeslaT4',
            'dataset_sources': [], 'competition_sources': [B.COMP],
            'kernel_sources': [B.V4_KERNEL], 'model_sources': []}
    json.dump(meta, open(os.path.join(d, 'kernel-metadata.json'), 'w'), indent=2)
    open(os.path.join(d, 'seeds_source.py'), 'w').write(src)
    print(f'{bb} -> kaggle/seeds_{bb}/{fn}  ({len(out)} cells)  id={meta["id"]}')


if __name__ == '__main__':
    for bb in FAM:
        build(bb)
