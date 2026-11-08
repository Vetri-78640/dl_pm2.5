"""Build the two ablation notebooks (H1-H4 of t9): study.py followed by ablations.py,
each running two single-change variants (seed 42, phase A recipe).
Output: kaggle/ablations_<part>/<slug>.ipynb and kernel-metadata.json.
"""
import json, os, re
import build_notebooks as B

PARTS = {'a': ['afnov4_allskip', 'afnov4_nosparse'], 'b': ['afnov4_pad', 'resgru_norec']}
HEADER = [
    ('# | seed | 42, one run |', '# | seed | 42, one run per variant (section 10) |'),
    ('# **Reference (not retrained):**', '# **Round-one reference (not retrained):**'),
    ('# **Decision rule (fixed before training):**', '# **Round-one decision rule:**'),
]


def source(variants):
    s = open(B.SRC).read()
    for a, b in HEADER:
        assert s.count(a) == 1, a
        s = s.replace(a, b)
    tail = "\n\nif __name__ == '__main__':\n    main()\n"
    assert s.endswith(tail) and "BACKBONE = 'resgru'" in s
    abl = open(os.path.join(B.HERE, '..', 'src', 'ablations.py')).read()
    assert abl.count('VARIANTS = []') == 1
    return s[:-len(tail)] + '\n\n' + abl.replace('VARIANTS = []', f'VARIANTS = {variants}')


def build(part, variants):
    title = f'Code4CleanAir Backbone Study - Ablations {part.upper()}'
    assert len(title) <= 50, title
    summary = (f'Single-change variants {", ".join(variants)} (seed 42, phase A recipe), testing why '
               'ResGRU-UNet beats AFNO V4 (sections 9 and 10).')
    arch = B.MODELS['resgru']['arch']
    notes = '**Networks:** AFNO V4 (section 4), ResGRU-UNet (below), and the variants in section 9.\n# \n# ' + arch
    src = source(variants)
    out = []
    for kind, text in B.cells_of(src):
        if kind == 'code' and text.startswith('# BACKBONE-CELL:'):
            first, _, rest = text.partition('\n')
            if 'resgru' not in first.split(':', 1)[1].split():
                continue
            text = rest
        if kind == 'markdown':
            text = (text.replace('{{TITLE}}', f'ablations {part.upper()}').replace('{{SUMMARY}}', summary)
                    .replace('{{ARCH}}', arch).replace('{{MODEL_NOTES}}', notes))
        out.append(B.to_nb_cell(kind, text))
    code = '\n'.join(''.join(c['source']) for c in out if c['cell_type'] == 'code')
    assert f'VARIANTS = {variants}' in code and 'class ResGRUNoRec' in code and '{{' not in json.dumps(out)
    compile(code, f'ablations_{part}', 'exec')
    nb = {'cells': out,
          'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                       'language_info': {'name': 'python', 'version': '3.10'}},
          'nbformat': 4, 'nbformat_minor': 4}
    d = os.path.join(B.HERE, '..', 'notebooks', 'kaggle', f'ablations_{part}')
    os.makedirs(d, exist_ok=True)
    fn = f'{B.slug(title)}.ipynb'
    json.dump(nb, open(os.path.join(d, fn), 'w'), indent=1)
    meta = {'id': f'{B.USER}/{B.slug(title)}', 'title': title, 'code_file': fn,
            'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
            'enable_gpu': True, 'enable_tpu': False, 'enable_internet': False,
            'machine_shape': 'NvidiaTeslaT4', 'dataset_sources': [], 'competition_sources': [B.COMP],
            'kernel_sources': [B.V4_KERNEL], 'model_sources': []}
    json.dump(meta, open(os.path.join(d, 'kernel-metadata.json'), 'w'), indent=2)
    open(os.path.join(d, 'ablations_source.py'), 'w').write(src)
    print(f'ablations {part} -> kaggle/ablations_{part}/{fn}  ({len(out)} cells)  id={meta["id"]}')


if __name__ == '__main__':
    for part, variants in PARTS.items():
        build(part, variants)
