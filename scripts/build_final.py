"""Build the final-round notebook: ResGRU-UNet seed 1337 + 2-seed comparison
with AFNO V4. Source: study.py (seed set to 1337) followed by final_round.py.
Output: kaggle/final/<slug>.ipynb and kernel-metadata.json.
"""
import json, os, re
import build_notebooks as B

ROUND1 = 'imvetri0/code4cleanair-backbone-study-resgru-unet'
TITLE = 'Code4CleanAir Backbone Study - Final Round'
HEADER = [
    ('# | seed | 42, one run |',
     '# | seed | 1337 trained here; seed 42 reused from round one (same code, same settings) |'),
    ('# | post-processing | none (no calibration, ensemble, TTA or blend) |',
     '# | post-processing | round one: none; final round (section 9): 2-seed mean, then V4\'s per-horizon calibration |'),
    ('# **Reference (not retrained):**', '# **Round-one reference (not retrained):**'),
    ('# **Decision rule (fixed before training):**', '# **Round-one decision rule:**'),
]


def source():
    s = open(B.SRC).read()
    for a, b in HEADER:
        assert s.count(a) == 1, a
        s = s.replace(a, b)
    s = re.sub(r'^SEED = 42 ', 'SEED = 1337 ', s, count=1, flags=re.M)
    tail = "\n\nif __name__ == '__main__':\n    main()\n"
    assert s.endswith(tail)
    return s[:-len(tail)] + '\n\n' + open(os.path.join(B.HERE, '..', 'src', 'final_round.py')).read()


def build():
    m = dict(B.MODELS['resgru'])
    m.update(title='final round - ResGRU-UNet vs AFNO V4, 2 seeds',
             summary=('Round one trained every backbone once (seed 42) in the AFNO V4 pipeline; '
                      'ResGRU-UNet was the only one better than AFNO V4. This notebook is the '
                      'final: it trains ResGRU-UNet seed 1337 with the round-one code, then gives '
                      'both finalists V4\'s full submitted recipe (2-seed mean + per-horizon '
                      'calibration) and scores them on the same validation windows (section 9).'))
    src = source()
    out = []
    lines = ['**ResGRU-UNet in this study**', '', m['arch'], '',
             'Changes from the original, all needed to put it in the shared pipeline:', '']
    lines += [f'- {c}' for c in m['changes']] + ['']
    lines += B.COMMON.format(**m).strip('\n').split('\n')
    notes = '\n# '.join(lines)
    for kind, text in B.cells_of(src):
        if kind == 'code' and text.startswith('# BACKBONE-CELL:'):
            first, _, rest = text.partition('\n')
            if 'resgru' not in first.split(':', 1)[1].split():
                continue
            text = rest
        if kind == 'markdown':
            text = (text.replace('{{TITLE}}', m['title']).replace('{{SUMMARY}}', m['summary'])
                    .replace('{{ARCH}}', m['arch']).replace('{{MODEL_NOTES}}', notes))
        out.append(B.to_nb_cell(kind, text))
    code = '\n'.join(''.join(c['source']) for c in out if c['cell_type'] == 'code')
    assert "BACKBONE = 'resgru'" in code and "SEED = 1337 " in code and '{{' not in json.dumps(out)
    compile(code, 'final', 'exec')
    nb = {'cells': out,
          'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                                      'name': 'python3'},
                       'language_info': {'name': 'python', 'version': '3.10'}},
          'nbformat': 4, 'nbformat_minor': 4}
    d = os.path.join(B.HERE, '..', 'notebooks', 'kaggle', 'final')
    os.makedirs(d, exist_ok=True)
    fn = f'{B.slug(TITLE)}.ipynb'
    json.dump(nb, open(os.path.join(d, fn), 'w'), indent=1)
    meta = {'id': f'{B.USER}/{B.slug(TITLE)}', 'title': TITLE, 'code_file': fn,
            'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
            'enable_gpu': True, 'enable_tpu': False, 'enable_internet': False,
            'machine_shape': 'NvidiaTeslaT4',
            'dataset_sources': [], 'competition_sources': [B.COMP],
            'kernel_sources': [B.V4_KERNEL, ROUND1], 'model_sources': []}
    json.dump(meta, open(os.path.join(d, 'kernel-metadata.json'), 'w'), indent=2)
    open(os.path.join(d, 'final_source.py'), 'w').write(src)
    print(f'final -> kaggle/final/{fn}  ({len(out)} cells)  id={meta["id"]}')


if __name__ == '__main__':
    build()
