"""Build the regression-metrics notebook: study.py (every network, all backbone cells) +
the ablation variant classes (ablations.py without its run section) + the analysis
helpers (analysis.py without its main call) + regression.py. No training.
Output: kaggle/regression/<slug>.ipynb, kernel-metadata.json, regression_source.py.
"""
import json, os
import build_notebooks as B

TITLE = 'Code4CleanAir Backbone Study - Regression Metrics'
SOURCES = [B.V4_KERNEL,
           'imvetri0/code4cleanair-backbone-study-resgru-unet',
           'imvetri0/code4cleanair-backbone-study-fmg-waclu',
           'imvetri0/code4cleanair-backbone-study-convlstm',
           'imvetri0/code4cleanair-backbone-study-patch-afno',
           'imvetri0/code4cleanair-backbone-study-final-round',
           'imvetri0/code4cleanair-backbone-study-seeds-1-3-resgru',
           'imvetri0/code4cleanair-backbone-study-seeds-1-3-afno-v4',
           'imvetri0/code4cleanair-backbone-study-ablations-a',
           'imvetri0/code4cleanair-backbone-study-ablations-b',
           'imvetri0/code4cleanair-backbone-study-5-seed-analysis']
HEADER = [
    ('# | seed | 42, one run |',
     '# | seeds | every model of the study, loaded from earlier runs (no training here) |'),
    ('# | post-processing | none (no calibration, ensemble, TTA or blend) |',
     '# | post-processing | 2- and 5-seed means; V4\'s per-horizon calibration (section 11) |'),
    ('# **Reference (not retrained):**', '# **Round-one reference (not retrained):**'),
    ('# **Decision rule (fixed before training):**', '# **Round-one decision rule:**'),
]


def cut(text, marker):
    assert text.count(marker) == 1, marker[:40]
    return text[:text.index(marker)].rstrip('\n')


def source():
    s = open(B.SRC).read()
    for a, b in HEADER:
        assert s.count(a) == 1, a
        s = s.replace(a, b)
    s = cut(s, "\n\nif __name__ == '__main__':\n    main()\n")
    abl = cut(open(os.path.join(B.HERE, '..', 'src', 'ablations.py')).read(), '# %% [markdown]\n# ## 10. Run the variants')
    ana = cut(open(os.path.join(B.HERE, '..', 'src', 'analysis.py')).read(), "\n\nif __name__ == '__main__':\n    analyze()")
    reg = open(os.path.join(B.HERE, '..', 'src', 'regression.py')).read()
    return '\n\n'.join([s, abl, ana, reg])


def build():
    assert len(TITLE) <= 50, TITLE
    summary = ('MAE, RMSE and R² (ug/m3, also on log1p) for every model the backbone study compared, on the same '
               '214 validation windows: round one, the 2-seed final round, the 5-seed extension, the 4 ablations '
               'and persistence. Cross-checked against the earlier competition-metric results (section 11).')
    arch = ('All five networks of the study (sections 4 and 4b) and the four ablation variants (section 9) are '
            'defined below. No network is trained here.')
    src = source()
    out = []
    for kind, text in B.cells_of(src):
        if kind == 'code' and text.startswith('# BACKBONE-CELL:'):
            text = text.partition('\n')[2]
        if kind == 'markdown':
            text = (text.replace('{{TITLE}}', 'regression metrics for every model').replace('{{SUMMARY}}', summary)
                    .replace('{{ARCH}}', arch).replace('{{MODEL_NOTES}}', arch))
        out.append(B.to_nb_cell(kind, text))
    code = '\n'.join(''.join(c['source']) for c in out if c['cell_type'] == 'code')
    for need in ('def regression', 'class PatchAFNO', 'class PM25ForecastNet', 'class FMGWACLUNet', 'class ResGRUNoRec',
                 'def fit_scales'):
        assert need in code, need
    assert 'run_variants()\n' not in code.replace('def run_variants():\n', '') and 'analyze()\n' not in code.replace('def analyze():\n', '')
    assert '{{' not in json.dumps(out)
    compile(code, 'regression', 'exec')
    nb = {'cells': out,
          'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                       'language_info': {'name': 'python', 'version': '3.10'}},
          'nbformat': 4, 'nbformat_minor': 4}
    d = os.path.join(B.HERE, '..', 'notebooks', 'kaggle', 'regression')
    os.makedirs(d, exist_ok=True)
    fn = f'{B.slug(TITLE)}.ipynb'
    json.dump(nb, open(os.path.join(d, fn), 'w'), indent=1)
    meta = {'id': f'{B.USER}/{B.slug(TITLE)}', 'title': TITLE, 'code_file': fn,
            'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
            'enable_gpu': True, 'enable_tpu': False, 'enable_internet': False,
            'machine_shape': 'NvidiaTeslaT4', 'dataset_sources': [], 'competition_sources': [B.COMP],
            'kernel_sources': SOURCES, 'model_sources': []}
    json.dump(meta, open(os.path.join(d, 'kernel-metadata.json'), 'w'), indent=2)
    open(os.path.join(d, 'regression_source.py'), 'w').write(src)
    print(f'regression -> kaggle/regression/{fn}  ({len(out)} cells)  id={meta["id"]}')


if __name__ == '__main__':
    build()
