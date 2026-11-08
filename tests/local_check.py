"""Local checks for the backbone study (CPU, no data needed).

1. Rebuild each original network from its original notebook and check the
   parameter count against the count printed in its Kaggle log.
2. Build the port with the original input width, copy the original weights
   into it and check that both give the same output (eval mode, random input,
   weights perturbed so the zero-initialized heads are not trivially equal).
3. Build each port with the study's 20 input channels: parameter count,
   output shape, persistence at init, finite gradients, and an autocast pass.

Usage: python tests/local_check.py path/to/raw (the archived original notebooks)
"""
import ast, json, os, sys, tempfile, importlib.util, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
if len(sys.argv) < 2:
    sys.exit(__doc__)
RAW = sys.argv[1]
V4_PY = os.path.join(HERE, '..', 'reference', 'afno_v4', 'afno_v4.py')

SOURCES = {
    'resgru': dict(
        nb='kaggle_archive/30_Res-1_-_v1/notebook.ipynb',
        names=['Config', 'ConvBlock', 'ConvGRUCell', 'ResGRUUNet'],
        logged=7_096_513),
    'fmgwaclu': dict(
        nb='kaggle_archive/13_notebookce92d325af_-_Version_2/notebook.ipynb',
        names=['Config', 'FMGConfig', 'ConvBlock', 'ConvLSTMCell', 'ConvGRUCell',
               'WindPhysicsModule', 'MetForecaster', 'FMGWACLUNet'],
        logged=3_451_057),
    'convlstm': dict(
        nb='kaggle_api/kernels/phase2_unsubmitted/convlstm-v2/convlstm-v2.ipynb',
        names=['T_IN', 'T_OUT', 'H', 'W', 'MET_VARS', 'EMIS_VARS', 'ALL_VARS', 'F_IN',
               'BASE_CHANNELS', 'DROPOUT_RATE', 'STOCHASTIC_DEPTH', 'DEVICE',
               'AdvectionFeature', 'StochasticDepth', 'ConvBnGelu', 'DoubleConv',
               'SpectralBlock2D', 'ConvLSTMCell', 'ConvLSTMEncoder', 'PM25ForecastNet'],
        logged=19_063_792),
    'patchafno': dict(
        nb='kaggle_archive/35_Adaptive_Fourier_Neural_Operator_-_50_epoch_gpu/notebook.ipynb',
        names=['Config', 'PatchEmbed', 'AdaptiveFourierBlock', 'TemporalProjectionHead',
               'AFNO_PM25'],
        logged=5_422_080),
}

FAIL = []


def check(ok, msg):
    print(('  PASS ' if ok else '  FAIL ') + msg)
    if not ok:
        FAIL.append(msg)


def load_original(key, fix=None):
    """Execute only the named top-level classes/assignments of a notebook.
    fix: optional (old, new) source replacement, applied exactly once."""
    spec = SOURCES[key]
    nb = json.load(open(os.path.join(RAW, spec['nb'])))
    code = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
    if fix:
        assert code.count(fix[0]) == 1, fix[0]
        code = code.replace(*fix)
    code = '\n'.join(l for l in code.split('\n') if not l.lstrip().startswith(('!', '%')))
    tree = ast.parse(code)
    ns = dict(os=os, math=math, np=np, torch=torch, nn=nn, F=F)
    want = set(spec['names'])
    for node in tree.body:
        name = None
        if isinstance(node, ast.ClassDef):
            name = node.name
        elif isinstance(node, ast.Assign):
            t = node.targets[0]
            if isinstance(t, ast.Name):
                name = t.id
            elif isinstance(t, ast.Tuple) and all(isinstance(e, ast.Name) for e in t.elts):
                name = t.elts[0].id
        if name in want:
            exec(compile(ast.Module([node], []), key, 'exec'), ns)
    missing = want - set(ns)
    assert not missing, f'{key}: not found in notebook: {missing}'
    return ns


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def perturb(m, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.02 * torch.randn(p.shape, generator=g))


def convblock_names(sd):
    """Same layers, different attribute names: original ConvBlock conv1/gn1/conv2/gn2
    vs V4's c1/g1/c2/g2, and notebook 13's ConvGRUCell conv_cand/gn_g/gn_c vs notebook 30's."""
    ren = {'conv1': 'c1', 'gn1': 'g1', 'conv2': 'c2', 'gn2': 'g2',
           'conv_cand': 'conv_candidate', 'gn_g': 'gn_gates', 'gn_c': 'gn_cand'}
    return {'.'.join(ren.get(t, t) for t in k.split('.')): v for k, v in sd.items()}


def max_diff(a, b):
    return (a - b).abs().max().item()


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


os.chdir(tempfile.mkdtemp())
torch.manual_seed(0)
S = import_file('study', os.path.join(HERE, '..', 'src', 'study.py'))
H, W, T = 140, 124, 10

print('=' * 70)
print('1-2. originals: logged parameter count, and port == original')
print('=' * 70)

# ResGRU-UNet (notebook 30)
ns = load_original('resgru')
cin = ns['Config'].NUM_INPUT_CHANNELS
orig = ns['ResGRUUNet']()
print(f'resgru: original input channels per hour = {cin}')
check(n_params(orig) == SOURCES['resgru']['logged'],
      f'resgru original params {n_params(orig):,} == logged {SOURCES["resgru"]["logged"]:,}')
port = S.ResGRUUNet(cin)
check(n_params(port) == n_params(orig), f'resgru port params {n_params(port):,} == original')
perturb(orig); port.load_state_dict(convblock_names(orig.state_dict()), strict=True)
x, last = torch.randn(1, cin, T, H, W), torch.randn(1, H, W)
orig.eval(); port.eval()
with torch.no_grad():
    d = max_diff(orig(x, last), port(x, last))
check(d < 1e-4, f'resgru port output == original (max diff {d:.2e})')

# FMG-WACLU (notebook 13)
ns = load_original('fmgwaclu')
cin = ns['Config'].NUM_INPUT_CHANNELS
orig = ns['FMGWACLUNet']()
print(f'fmgwaclu: original input channels per hour = {cin}')
check(n_params(orig) == SOURCES['fmgwaclu']['logged'],
      f'fmgwaclu original params {n_params(orig):,} == logged {SOURCES["fmgwaclu"]["logged"]:,}')
port = S.FMGWACLUNet(cin)
check(n_params(port) == n_params(orig), f'fmgwaclu port params {n_params(port):,} == original')
perturb(orig); port.load_state_dict(convblock_names(orig.state_dict()), strict=True)
x, last = torch.randn(1, cin, T, H, W), torch.randn(1, H, W)
orig.eval(); port.eval()
with torch.no_grad():
    d = max_diff(orig(x, last), port(x, last))
check(d < 1e-4, f'fmgwaclu port output == original (max diff {d:.2e})')

# ConvLSTM (convlstm-v2)
ns = load_original('convlstm')
cin = len(ns['ALL_VARS'])
orig = ns['PM25ForecastNet']()
print(f'convlstm: original input channels per hour = {cin}')
check(n_params(orig) == SOURCES['convlstm']['logged'],
      f'convlstm original params {n_params(orig):,} == logged {SOURCES["convlstm"]["logged"]:,}')
port = S.PM25ForecastNet(cin)
season = sum(p.numel() for k, p in orig.named_parameters() if k.split('.')[0] in
             ('s_emb', 'film_sc', 'film_bi'))
check(n_params(port) == n_params(orig) - season,
      f'convlstm port params {n_params(port):,} == original minus season FiLM ({season:,})')
perturb(orig)
with torch.no_grad():
    for k in ('film_sc', 'film_bi'):
        getattr(orig, k).weight.zero_(); getattr(orig, k).bias.zero_()
res = port.load_state_dict(orig.state_dict(), strict=False)
check(not res.missing_keys and set(k.split('.')[0] for k in res.unexpected_keys)
      <= {'s_emb', 'film_sc', 'film_bi'},
      f'convlstm port has every original weight except season FiLM '
      f'(missing {res.missing_keys}, extra {sorted(set(k.split(".")[0] for k in res.unexpected_keys))})')
x = torch.randn(1, cin, T, H, W)
orig.eval(); port.eval()
with torch.no_grad():
    d = max_diff(orig(x, torch.zeros(1, dtype=torch.long)), port(x, torch.zeros(1, H, W)))
check(d < 1e-4, f'convlstm port output == original with FiLM set to identity (max diff {d:.2e})')

# patch AFNO (notebook 35)
ns = load_original('patchafno')
cfg = ns['Config']
cin = cfg.NUM_FEATURES
orig = ns['AFNO_PM25'](cfg)
print(f'patchafno: original input channels per hour = {cin}')
check(n_params(orig) == SOURCES['patchafno']['logged'],
      f'patchafno original params {n_params(orig):,} == logged {SOURCES["patchafno"]["logged"]:,}')
port = S.PatchAFNO(cin, dim=cfg.EMBED_DIM, depth=cfg.DEPTH, heads=cfg.NUM_HEADS,
                   mlp_ratio=cfg.MLP_RATIO, drop=cfg.DROP_RATE, patch=cfg.PATCH_SIZE)
check(n_params(port) == n_params(orig), f'patchafno port params {n_params(port):,} == original')
perturb(orig); port.load_state_dict(orig.state_dict(), strict=True)
x = torch.randn(1, cin, T, H, W)
orig.eval(); port.eval()
with torch.no_grad():
    d_bug = max_diff(orig(x), port(x, torch.zeros(1, H, W)))
print(f'  info: original as written differs from the port by {d_bug:.2e} '
      f'(irfft2(xf.float()) keeps only the real part of the spectrum)')
FIX = ('torch.fft.irfft2(xf.float(), s=(Hp, Wp)', 'torch.fft.irfft2(xf, s=(Hp, Wp)')
fixed = load_original('patchafno', FIX)['AFNO_PM25'](cfg)
fixed.load_state_dict(orig.state_dict(), strict=True)
fixed.eval()
with torch.no_grad():
    d = max_diff(fixed(x), port(x, torch.zeros(1, H, W)))
check(d < 1e-4, f'patchafno port output == original with only xf.float() -> xf (max diff {d:.2e})')
check(d_bug > 100 * max(d, 1e-7), 'patchafno: the xf.float() line is the only difference')

# AFNO V4 (afno_v4.py)
V4 = import_file('afno_v4', V4_PY)
orig, port = V4.AFNOv4(), S.AFNOv4()
check(n_params(port) == n_params(orig), f'afnov4 params {n_params(port):,} == afno_v4.py')
perturb(orig); port.load_state_dict(orig.state_dict(), strict=True)
x, last = torch.randn(1, 20, T, H, W), torch.randn(1, H, W)
orig.eval(); port.eval()
with torch.no_grad():
    d = max_diff(orig(x, last), port(x, last))
check(d < 1e-4, f'afnov4 copy output == afno_v4.py (max diff {d:.2e})')

print('=' * 70)
print('3. study configuration (20 input channels)')
print('=' * 70)
amp_dev, amp_dt = 'cpu', torch.float16   # same dtype flow as CUDA fp16 autocast; MPS lacks grid_sample border mode
counts = {}
for key in ['afnov4', 'convlstm', 'fmgwaclu', 'resgru', 'patchafno']:
    torch.manual_seed(42)
    m = S.BACKBONES[key]()
    counts[key] = n_params(m)
    x, last = torch.randn(2, 20, T, H, W), torch.randn(2, H, W)
    m.eval()
    with torch.no_grad():
        y = m(x, last)
    check(tuple(y.shape) == (2, 16, H, W), f'{key}: output shape {tuple(y.shape)}')
    check(max_diff(y, last[:, None].expand_as(y)) < 1e-5, f'{key}: equals persistence at init')
    perturb(m)
    m.train()
    y = m(x, last)
    ((y - last[:, None]) ** 2).mean().backward()
    g = [p.grad for p in m.parameters() if p.requires_grad]
    check(all(t is not None and torch.isfinite(t).all() for t in g),
          f'{key}: every parameter gets a finite gradient')
    m.zero_grad(set_to_none=True)
    with torch.autocast(amp_dev, dtype=amp_dt):
        y = m(x[:1].to(amp_dev), last[:1].to(amp_dev))
    loss = ((y.float() - last[:1, None].to(amp_dev)) ** 2).mean()
    loss.backward()
    check(torch.isfinite(loss).item(), f'{key}: autocast ({amp_dev}, {amp_dt}) forward + backward runs')
    print(f'  {key}: {counts[key]:,} parameters')
    del m

print('=' * 70)
print(json.dumps(counts))
print('ALL CHECKS PASSED' if not FAIL else f'{len(FAIL)} CHECK(S) FAILED')
sys.exit(1 if FAIL else 0)
