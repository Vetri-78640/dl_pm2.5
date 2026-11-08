"""Build the 5-seed analysis notebook: study.py (shared pipeline, both networks)
followed by analysis.py. No training. Output: kaggle/analysis/<slug>.ipynb.
"""
import json, os, re
import build_notebooks as B

TITLE = 'Code4CleanAir Backbone Study - 5-Seed Analysis'
SOURCES = [B.V4_KERNEL, 'imvetri0/code4cleanair-backbone-study-resgru-unet',
           'imvetri0/code4cleanair-backbone-study-final-round',
           'imvetri0/code4cleanair-backbone-study-seeds-1-3-resgru',
           'imvetri0/code4cleanair-backbone-study-seeds-1-3-afno-v4']
HEADER = [
    ('# | seed | 42, one run |',
     '# | seeds | 42, 1337, 1, 2, 3 per finalist, loaded from earlier runs (no training here) |'),
    ('# | post-processing | none (no calibration, ensemble, TTA or blend) |',
     '# | post-processing | 5-seed means; V4\'s per-horizon calibration in and out of sample (section 9) |'),
    ('# **Reference (not retrained):**', '# **Round-one reference (not retrained):**'),
    ('# **Decision rule (fixed before training):**', '# **Round-one decision rule:**'),
]


def source():
    s = open(B.SRC).read()
    for a, b in HEADER:
        assert s.count(a) == 1, a
        s = s.replace(a, b)
    tail = "\n\nif __name__ == '__main__':\n    main()\n"
    assert s.endswith(tail) and "BACKBONE = 'resgru'" in s
    return s[:-len(tail)] + '\n\n' + open(os.path.join(B.HERE, '..', 'src', 'analysis.py')).read()


def build():
    assert len(TITLE) <= 50, TITLE
    summary = ('Final analysis of the backbone study: 5 seeds each of ResGRU-UNet and AFNO V4, scored on '
               'the same validation windows, with block-bootstrap uncertainty, breakdowns by lead hour, '
               'month, concentration and location, calibration in and out of sample, and cost (section 9).')
    arch = B.MODELS['resgru']['arch']
    notes = '**Networks:** ResGRU-UNet (below) and AFNO V4 (section 4).\n# \n# ' + arch
    src = source()
    out = []
    for kind, text in B.cells_of(src):
        if kind == 'code' and text.startswith('# BACKBONE-CELL:'):
            first, _, rest = text.partition('\n')
            if 'resgru' not in first.split(':', 1)[1].split():
                continue
            text = rest
        if kind == 'markdown':
            text = (text.replace('{{TITLE}}', '5-seed analysis, ResGRU-UNet vs AFNO V4').replace('{{SUMMARY}}', summary)
                    .replace('{{ARCH}}', arch).replace('{{MODEL_NOTES}}', notes))
        out.append(B.to_nb_cell(kind, text))
    code = '\n'.join(''.join(c['source']) for c in out if c['cell_type'] == 'code')
    assert 'def analyze' in code and 'class ResGRUUNet' in code and '{{' not in json.dumps(out)
    compile(code, 'analysis', 'exec')
    nb = {'cells': out,
          'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                       'language_info': {'name': 'python', 'version': '3.10'}},
          'nbformat': 4, 'nbformat_minor': 4}
    d = os.path.join(B.HERE, '..', 'notebooks', 'kaggle', 'analysis')
    os.makedirs(d, exist_ok=True)
    fn = f'{B.slug(TITLE)}.ipynb'
    json.dump(nb, open(os.path.join(d, fn), 'w'), indent=1)
    meta = {'id': f'{B.USER}/{B.slug(TITLE)}', 'title': TITLE, 'code_file': fn,
            'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
            'enable_gpu': True, 'enable_tpu': False, 'enable_internet': False,
            'machine_shape': 'NvidiaTeslaT4', 'dataset_sources': [], 'competition_sources': [B.COMP],
            'kernel_sources': SOURCES, 'model_sources': []}
    json.dump(meta, open(os.path.join(d, 'kernel-metadata.json'), 'w'), indent=2)
    open(os.path.join(d, 'analysis_source.py'), 'w').write(src)
    print(f'analysis -> kaggle/analysis/{fn}  ({len(out)} cells)  id={meta["id"]}')


if __name__ == '__main__':
    build()
