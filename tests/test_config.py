"""configs/study.yaml must match the settings the Kaggle runs used (src/study.py and the stage sources)."""
import ast
import os
import re

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = yaml.safe_load(open(os.path.join(ROOT, 'configs', 'study.yaml')))


def assignments(path, cls=None):
    """Literal assignments in a source file, optionally inside one class."""
    tree = ast.parse(open(os.path.join(ROOT, path)).read())
    body = tree.body
    if cls:
        body = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body
    out = {}
    for node in body:
        if not isinstance(node, ast.Assign):
            continue
        v = node.value
        try:
            if isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == 'dict':
                value = {k.arg: ast.literal_eval(k.value) for k in v.keywords}
            else:
                value = ast.literal_eval(v)
        except ValueError:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                out[target.id] = value
            elif isinstance(target, ast.Tuple) and isinstance(value, tuple):
                for t, v in zip(target.elts, value):
                    out[t.id] = v
    return out


C = assignments('src/study.py', 'C')
SEED = assignments('src/study.py')['SEED']
ANA = assignments('src/analysis.py')
RIDGE = assignments('src/ridge.py')


def test_data():
    d = CFG['data']
    assert d['months'] == C['MONTHS'] and d['features'] == C['FEATS']
    assert set(d['log_features']) == C['LOG_FEATS']
    assert d['grid'] == [C['H'], C['W']]
    assert (d['hours_in'], d['hours_out'], d['window']) == (C['T_IN'], C['T_OUT'], C['WIN'])
    assert d['test_windows'] == C['N_TEST'] and d['input_channels'] == C['C_IN'] and d['clip_inputs'] == C['CLIP_IN']


def test_split_and_episodes():
    s = CFG['split']
    assert (s['val_fraction'], s['gap_hours'], s['val_stride']) == (C['VAL_FRAC'], C['GAP'], C['VAL_STRIDE'])
    e = CFG['episodes']
    assert e['stl_robust'] == C['STL_ROBUST']
    src = open(os.path.join(ROOT, 'src', 'study.py')).read()
    assert re.search(r'STL\(ts, period=%d' % e['stl_period'], src)
    assert f"(r > {e['sigma']} * s) & (ts > {int(e['min_value'])})" in src


def test_afno_v4():
    a = CFG['afno_v4']
    assert a['encoder'] == C['ENC']
    assert (a['afno_blocks'], a['mid_afno_blocks'], a['post_afno_blocks']) == (C['AFNO_BLOCKS'], C['MID_AFNO_BLOCKS'], C['POST_AFNO_BLOCKS'])
    assert (a['afno_heads'], a['mlp_ratio'], a['dropout'], a['translator_layers']) == (C['AFNO_HEADS'], C['MLP_RATIO'], C['DROP'], C['TRANS_LAYERS'])


def test_training():
    t = CFG['training']
    assert (t['epochs'], t['learning_rate'], t['batch'], t['patience']) == (C['A_EPOCHS'], C['A_LR'], C['A_BATCH'], C['A_PATIENCE'])
    assert (t['weight_decay'], t['ema_decay'], t['episode_weight']) == (C['WD'], C['EMA_DECAY'], C['EP_WT_A'])
    assert t['loss'] == C['LOSS_A'] and t['horizon_weights'] == [C['HW_MIN'], C['HW_MAX']]
    assert t['seed'] == SEED


def test_evaluation_and_ridge():
    e = CFG['evaluation']
    assert e['seeds'] == ANA['SEEDS5']
    assert (e['bootstrap_block'], e['bootstrap_resamples']) == (ANA['BLOCK'], ANA['N_BOOT'])
    assert tuple(e['calibration_range']) == ANA['CAL_RANGE']
    verdict = open(os.path.join(ROOT, 'scripts', 'ablation_verdict.py')).read()
    assert f"max(2 * s, {e['ablation_threshold']})" in verdict
    r = CFG['ridge']
    assert r['lambdas'] == RIDGE['RIDGE_LAMBDAS'] and r['cell_stride'] == RIDGE['RIDGE_STRIDE']
