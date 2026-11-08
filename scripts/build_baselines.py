"""Build the CPU baselines notebook: the regression notebook's source without its main call,
plus baselines.py. Output: kaggle/baselines/<slug>.ipynb, kernel-metadata.json, baselines_source.py.
"""
import json, os
import build_notebooks as B
import build_regression as R

TITLE = 'Code4CleanAir Backbone Study - Baselines'
SOURCES = [B.V4_KERNEL, 'imvetri0/code4cleanair-backbone-study-regression-metrics']


def build():
    src = (R.cut(R.source(), "\n\nif __name__ == '__main__':\n    regression()") + '\n\n'
           + open(os.path.join(B.HERE, '..', 'src', 'baselines.py')).read())
    note = 'Persistence and seasonal-naive baselines on the val windows (section 12). No model is used.'
    out = []
    for kind, text in B.cells_of(src):
        if kind == 'code' and text.startswith('# BACKBONE-CELL:'):
            text = text.partition('\n')[2]
        if kind == 'markdown':
            text = (text.replace('{{TITLE}}', 'baselines').replace('{{SUMMARY}}', note)
                    .replace('{{ARCH}}', note).replace('{{MODEL_NOTES}}', note))
        out.append(B.to_nb_cell(kind, text))
    code = '\n'.join(''.join(c['source']) for c in out if c['cell_type'] == 'code')
    assert 'def baselines' in code and 'regression()\n' not in code.replace('def regression():\n', '')
    compile(code, 'baselines', 'exec')
    nb = {'cells': out, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                     'language_info': {'name': 'python', 'version': '3.10'}},
          'nbformat': 4, 'nbformat_minor': 4}
    d = os.path.join(B.HERE, '..', 'notebooks', 'kaggle', 'baselines')
    os.makedirs(d, exist_ok=True)
    fn = f'{B.slug(TITLE)}.ipynb'
    json.dump(nb, open(os.path.join(d, fn), 'w'), indent=1)
    meta = {'id': f'{B.USER}/{B.slug(TITLE)}', 'title': TITLE, 'code_file': fn, 'language': 'python',
            'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': False, 'enable_tpu': False,
            'enable_internet': False, 'dataset_sources': [], 'competition_sources': [B.COMP],
            'kernel_sources': SOURCES, 'model_sources': []}
    json.dump(meta, open(os.path.join(d, 'kernel-metadata.json'), 'w'), indent=2)
    open(os.path.join(d, 'baselines_source.py'), 'w').write(src)
    print(f'baselines -> kaggle/baselines/{fn}  id={meta["id"]}')


if __name__ == '__main__':
    build()
