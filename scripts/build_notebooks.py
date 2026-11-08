"""Build one Kaggle notebook per backbone from study.py (percent format).

Each notebook gets the shared pipeline, the AFNO V4 cell (pipeline check) and
only its own network cells. Output: kaggle/<backbone>/<slug>.ipynb and
kernel-metadata.json, ready for `kaggle kernels push -p kaggle/<backbone>`.
"""
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 'src', 'study.py')
USER = 'imvetri0'
COMP = 'anrf-aise-hack-phase-2-theme-2-pollution-forecasting-iitd'
V4_KERNEL = 'imvetri0/afno-v4'

COMMON = """
Verification and scope:

- Verified locally (`local_check.py`), with the original input width: {verified}.
- Not carried over (training recipe or post-processing, not architecture):
  {dropped}.
- Parameters in this study (20 input channels): {params}."""

MODELS = {
    'convlstm': dict(
        title='ConvLSTM',
        summary=('Network under test: **ConvLSTM** - `PM25ForecastNet` from the '
                 'convlstm-v2 kernel. No submission is confirmed for this code '
                 '(upload 31, private 0.7594, links to it, unconfirmed).'),
        arch=('A ConvLSTM reads the 10 input hours at full resolution (input '
              'channels plus a parameter-free wind-advected PM2.5 frame). Its '
              'state goes through a BatchNorm U-Net with stochastic depth and two '
              'FNO spectral blocks (8 x 8 modes) at the bottleneck. A conv head '
              'outputs all 16 hours at once.'),
        changes=['input: 20 channels per hour (original 16: PM2.5, 8 met, 7 emission fields)',
                 'the advection feature still shifts channel 0 (PM2.5) by channels 3/4 (u10/v10); '
                 'these are now per-pixel z-scores',
                 'season embedding + FiLM removed (66,560 parameters): the shared pipeline has no '
                 'season input',
                 'output: log1p(last frame) + head output, last conv zero-initialized (original '
                 'predicted the normalized field directly)',
                 'a duplicated `enc1`/`enc2` call whose result was discarded is not repeated: '
                 'same outputs, it only added BatchNorm running-stat updates in training'],
        verified=('every original weight except the season FiLM (19,063,792 - 66,560 = '
                  '18,997,232 parameters) and exactly the original output for the same '
                  'weights with FiLM set to identity'),
        dropped='batch 8, 50 epochs, its loss, 3-seed ensemble, TTA',
        params='19,001,840'),
    'fmgwaclu': dict(
        title='FMG-WACLU',
        summary=('Network under test: **FMG-WACLU** - `FMGWACLUNet` from notebook 13 '
                 '(submission 13, private 0.8465, with its own pipeline, a pinball '
                 'fine-tune and a blend).'),
        arch=('Wind physics channels (speed, direction sin/cos, divergence) feed a '
              'ConvLSTM over the 10 input hours. A second ConvLSTM (MetForecaster) '
              'rolls the input fields 16 hours forward. A learned gate fuses the '
              'history and forecast states. Then a U-Net encoder, and a ConvGRU '
              'decoder loop that emits one hour per step.'),
        changes=['input: 20 channels per hour (original 16); u10/v10 are channels 3/4 as in '
                 'the original',
                 'nothing else: the network already predicted log1p(last frame) + change with a '
                 'zero-initialized head'],
        verified=('the original parameter count (3,451,057) and exactly the original '
                  'output for the same weights'),
        dropped=('the 0.90 pinball fine-tune and the stable/spike blend; batch 2 '
                 '(here effective batch 4, with gradient accumulation if 4 does not fit)'),
        params='3,454,645'),
    'resgru': dict(
        title='ResGRU-UNet',
        summary=('Network under test: **ResGRU-UNet** - `ResGRUUNet` from notebook 30 '
                 '(submission 30, private 0.8506, with its own pipeline).'),
        arch=('The 10 input hours are folded into channels. A U-Net encoder '
              '(64/128/256) feeds a ConvGRU that is iterated 16 times at the '
              'bottleneck. Each step is decoded through the skip connections to '
              'one forecast hour.'),
        changes=['input: 20 x 10 channels (original 16 x 10)',
                 'nothing else: the network already predicted log1p(last frame) + change with a '
                 'zero-initialized head'],
        verified=('the original parameter count (7,096,513) and exactly the original '
                  'output for the same weights'),
        dropped='lr 1e-3, 30 epochs, its episode-weighted Huber loss and fine-tune chain',
        params='7,122,113'),
    'patchafno': dict(
        title='Patch AFNO',
        summary=('Network under test: **patch AFNO** - `AFNO_PM25` from notebook 35 '
                 '(submission 35, private 0.8218, with a random split).'),
        arch=('The 10 input hours are folded into channels and cut into 4 x 4 '
              'patches (35 x 31 tokens, dim 256). 8 adaptive Fourier blocks (FFT, '
              'per-head complex weights, soft-shrink, inverse FFT, MLP). A linear '
              'head unpatchifies each token to 16 hours x 4 x 4 pixels.'),
        changes=['input: 20 channels per hour (original 13)',
                 'output: log1p(last frame) + head output, last Linear zero-initialized (original '
                 'predicted the normalized field directly)',
                 'spectral bug fixed: the original called `irfft2(xf.float())`; `.float()` on a '
                 'complex tensor keeps only its real part, so the imaginary half of every filtered '
                 'spectrum was dropped (the warning is in its Kaggle log). Here the complex '
                 'spectrum is passed intact',
                 'the deprecated `torch.cuda.amp.autocast(enabled=False)` is replaced by '
                 '`torch.autocast(<device>, enabled=False)` around the whole spectral product'],
        verified=('the original parameter count (5,422,080) and exactly the output of '
                  'the original after the one-line `xf.float()` -> `xf` fix, for the same weights'),
        dropped='50 epochs, cosine schedule, its loss',
        params='5,708,800'),
}


def slug(title):
    return re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')


def cells_of(text):
    cells, kind, buf = [], None, []
    for line in text.split('\n'):
        if line.startswith('# %% [markdown]'):
            cells.append((kind, buf)); kind, buf = 'markdown', []
        elif line.startswith('# %%'):
            cells.append((kind, buf)); kind, buf = 'code', []
        else:
            buf.append(line)
    cells.append((kind, buf))
    return [(k, '\n'.join(b).strip('\n')) for k, b in cells if k and '\n'.join(b).strip()]


def to_nb_cell(kind, text):
    if kind == 'markdown':
        text = '\n'.join(l[2:] if l.startswith('# ') else l.lstrip('#')
                         for l in text.split('\n'))
    lines = [l + '\n' for l in text.split('\n')]
    lines[-1] = lines[-1].rstrip('\n')
    cell = {'cell_type': kind, 'metadata': {}, 'source': lines}
    if kind == 'code':
        cell.update(execution_count=None, outputs=[])
    return cell


def build(bb, src):
    m = MODELS[bb]
    title = f'Code4CleanAir Backbone Study - {m["title"]}'
    lines = [f'**{m["title"]} in this study**', '', m['arch'], '',
             'Changes from the original, all needed to put it in the shared pipeline:', '']
    lines += [f'- {c}' for c in m['changes']] + ['']
    lines += COMMON.format(**m).strip('\n').split('\n')
    notes = '\n# '.join(lines)
    out = []
    for kind, text in cells_of(src):
        if kind == 'code' and text.startswith('# BACKBONE-CELL:'):
            first, _, rest = text.partition('\n')
            if bb not in first.split(':', 1)[1].split():
                continue
            text = rest
        if kind == 'markdown':
            text = (text.replace('{{TITLE}}', m['title'])
                    .replace('{{SUMMARY}}', m['summary'])
                    .replace('{{ARCH}}', m['arch'])
                    .replace('{{MODEL_NOTES}}', notes))
        else:
            text = re.sub(r"^BACKBONE = '[a-z]+'", f"BACKBONE = '{bb}'", text, count=1, flags=re.M)
        out.append(to_nb_cell(kind, text))
    code = '\n'.join(''.join(c['source']) for c in out if c['cell_type'] == 'code')
    assert f"BACKBONE = '{bb}'" in code and '{{' not in json.dumps(out)
    compile(code, bb, 'exec')
    nb = {'cells': out,
          'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                                      'name': 'python3'},
                       'language_info': {'name': 'python', 'version': '3.10'}},
          'nbformat': 4, 'nbformat_minor': 4}
    d = os.path.join(HERE, '..', 'notebooks', 'kaggle', bb)
    os.makedirs(d, exist_ok=True)
    fn = f'{slug(title)}.ipynb'
    json.dump(nb, open(os.path.join(d, fn), 'w'), indent=1)
    meta = {'id': f'{USER}/{slug(title)}', 'title': title, 'code_file': fn,
            'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
            'enable_gpu': True, 'enable_tpu': False, 'enable_internet': False,
            'machine_shape': 'NvidiaTeslaT4',
            'dataset_sources': [], 'competition_sources': [COMP],
            'kernel_sources': [V4_KERNEL], 'model_sources': []}
    json.dump(meta, open(os.path.join(d, 'kernel-metadata.json'), 'w'), indent=2)
    print(f'{bb:10s} -> kaggle/{bb}/{fn}  ({len(out)} cells)  id={meta["id"]}')


if __name__ == '__main__':
    src = open(SRC).read()
    for bb in MODELS:
        build(bb, src)
