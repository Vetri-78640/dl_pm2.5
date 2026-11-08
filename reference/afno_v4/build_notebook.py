"""Convert afno_v4.py (percent format) into a Kaggle notebook."""
import json, os

SRC = os.path.join(os.path.dirname(__file__), 'afno_v4.py')
OUT = os.path.join(os.path.dirname(__file__), 'Code4CleanAir_Model4_AFNOv4.ipynb')

cells = []
cur_type, cur_lines = None, []

def flush():
    global cur_type, cur_lines
    if cur_type is None:
        return
    text = '\n'.join(cur_lines).strip('\n')
    if not text:
        cur_type, cur_lines = None, []
        return
    if cur_type == 'markdown':
        text = '\n'.join(l[2:] if l.startswith('# ') else (l[1:] if l.startswith('#') else l)
                         for l in text.split('\n'))
        cells.append({'cell_type': 'markdown', 'metadata': {},
                      'source': text.splitlines(keepends=False)})
    else:
        cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None,
                      'outputs': [], 'source': text.splitlines(keepends=False)})
    cells[-1]['source'] = [l + '\n' for l in cells[-1]['source']]
    if cells[-1]['source']:
        cells[-1]['source'][-1] = cells[-1]['source'][-1].rstrip('\n')
    cur_type, cur_lines = None, []

for line in open(SRC):
    line = line.rstrip('\n')
    if line.startswith('# %% [markdown]'):
        flush(); cur_type = 'markdown'
    elif line.startswith('# %%'):
        flush(); cur_type = 'code'
    else:
        if cur_type is None:
            cur_type = 'code'
        cur_lines.append(line)
flush()

nb = {
    'cells': cells,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                       'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.10'},
    },
    'nbformat': 4, 'nbformat_minor': 4,
}
json.dump(nb, open(OUT, 'w'), indent=1)
print(f'wrote {OUT}  ({len(cells)} cells)')
