"""Build the CPU ridge notebook: the regression source + baselines.py helpers (no main calls) + ridge.py.
Output: kaggle/ridge/<slug>.ipynb, kernel-metadata.json, ridge_source.py.
"""
import json, os
import build_notebooks as B
import build_regression as R

TITLE = 'Code4CleanAir Backbone Study - Ridge'
SOURCES = [B.V4_KERNEL]


def build():
    src = (R.cut(R.source(), "\n\nif __name__ == '__main__':\n    regression()") + '\n\n'
           + R.cut(open(os.path.join(B.HERE, '..', 'src', 'baselines.py')).read(), "\n\nif __name__ == '__main__':\n    baselines()") + '\n\n'
           + open(os.path.join(B.HERE, '..', 'src', 'ridge.py')).read())
    note = 'Ridge regression baseline on the val windows and the test set (section 13). No neural network is trained.'
    out = []
    for kind, text in B.cells_of(src):
        if kind == 'code' and text.startswith('# BACKBONE-CELL:'):
            text = text.partition('\n')[2]
        if kind == 'markdown':
            text = (text.replace('{{TITLE}}', 'baselines').replace('{{SUMMARY}}', note)
                    .replace('{{ARCH}}', note).replace('{{MODEL_NOTES}}', note))
        out.append(B.to_nb_cell(kind, text))
    code = '\n'.join(''.join(c['source']) for c in out if c['cell_type'] == 'code')
    assert 'def ridge' in code and 'baselines()\n' not in code.replace('def baselines():\n', '')
    compile(code, 'baselines', 'exec')
    nb = {'cells': out, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                     'language_info': {'name': 'python', 'version': '3.10'}},
          'nbformat': 4, 'nbformat_minor': 4}
    d = os.path.join(B.HERE, '..', 'notebooks', 'kaggle', 'ridge')
    os.makedirs(d, exist_ok=True)
    fn = f'{B.slug(TITLE)}.ipynb'
    json.dump(nb, open(os.path.join(d, fn), 'w'), indent=1)
    meta = {'id': f'{B.USER}/{B.slug(TITLE)}', 'title': TITLE, 'code_file': fn, 'language': 'python',
            'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': False, 'enable_tpu': False,
            'enable_internet': False, 'dataset_sources': [], 'competition_sources': [B.COMP],
            'kernel_sources': SOURCES, 'model_sources': []}
    json.dump(meta, open(os.path.join(d, 'kernel-metadata.json'), 'w'), indent=2)
    open(os.path.join(d, 'ridge_source.py'), 'w').write(src)
    print(f'ridge -> kaggle/ridge/{fn}  id={meta["id"]}')


if __name__ == '__main__':
    build()
