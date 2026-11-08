# %% [markdown]
# ## 9. Ablations: why does ResGRU-UNet beat AFNO V4?
#
# Each variant changes one thing in one finalist and is trained once (seed 42) with
# the same phase A recipe. Hypotheses, predictions and the decision rule were fixed
# before these runs.
#
# | variant | hypothesis | change | prediction if the hypothesis holds |
# |---|---|---|---|
# | `afnov4_allskip` | H1: full-resolution time information | AFNO V4 skips fuse all 10 hours (1x1 conv) instead of [last hour, mean] | AFNO V4 improves, mainly global SMAPE |
# | `afnov4_pad` | H2: spectral wrap-around | every AFNO block runs on a zero-padded 2H x 2W map, cropped back | AFNO V4 improves |
# | `afnov4_nosparse` | H3: soft-thresholding removes detail | AFNO soft-threshold 0.01 -> 0 | AFNO V4 improves, mainly global SMAPE |
# | `resgru_norec` | H4: lead-time recurrence | ResGRU-UNet's ConvGRU replaced by a per-lead embedding + 3 ConvBlocks (no recurrence, same parameter count within 0.1%) | ResGRU-UNet gets worse, mainly episode correlation |

# %%
class AFNO2DPad(AFNO2D):
    """AFNO2D on a map zero-padded to 2H x 2W, cropped back: opposite edges no longer wrap."""
    def forward(self, x):
        H, W = x.shape[-2:]
        return super().forward(F.pad(x, [0, W, 0, H]))[..., :H, :W]


def afno_variant(pad=False, sparsity=None):
    m = AFNOv4()
    for blk in m.modules():
        if isinstance(blk, AFNOBlock):
            if pad:
                old = blk.afno
                new = AFNO2DPad(old.nb * old.bs, old.nb, old.sparsity)
                new.load_state_dict(old.state_dict())
                blk.afno = new
            if sparsity is not None:
                blk.afno.sparsity = sparsity
    return m


def widen_fuse(old, T):
    """1x1 conv on all T hours that equals `old` on [last hour, mean] at init."""
    w = old.weight.data
    e = w.shape[0]
    nw = (w[:, e:] / T).repeat(1, T, 1, 1)
    nw[:, (T - 1) * e:] += w[:, :e]
    new = nn.Conv2d(T * e, e, 1, bias=False)
    new.weight.data.copy_(nw)
    return new


class AFNOv4AllSkip(AFNOv4):
    """AFNO V4 whose skip connections keep every input hour (H1).
    At init it computes exactly what AFNO V4 computes; training can then weight each hour."""
    def __init__(self, cfg=C):
        super().__init__(cfg)
        self.fuse0 = widen_fuse(self.fuse0, cfg.T_IN)
        self.fuse1 = widen_fuse(self.fuse1, cfg.T_IN)
        self.fuse2 = widen_fuse(self.fuse2, cfg.T_IN)

    def forward(self, x, last_log):
        B, Ch, T, H, W = x.shape
        xt = x.permute(0, 2, 1, 3, 4).reshape(B * T, Ch, H, W)
        e0 = self.enc0(xt)
        e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(self.pool(e1))
        for blk in self.mid_afno:
            e2 = blk(e2)
        bn = self.pool_bn(e2)
        for blk in self.afno:
            bn = blk(bn)

        def fuse(e, conv):
            return conv(e.reshape(B, T * e.shape[1], *e.shape[2:]))

        s0, s1, s2 = fuse(e0, self.fuse0), fuse(e1, self.fuse1), fuse(e2, self.fuse2)
        bn = bn.reshape(B, T, *bn.shape[1:])
        bn_out = self.trans(bn) + self.lead_emb.weight[None, :, :, None, None]
        To = self.cfg.T_OUT
        bo = bn_out.reshape(B * To, *bn_out.shape[2:])
        for blk in self.post_afno:
            bo = blk(bo)
        rep = lambda s: s.unsqueeze(1).expand(-1, To, -1, -1, -1).reshape(B * To, *s.shape[1:])
        d2 = self.dec2(torch.cat([self._pad(self.up2(bo), rep(s2)), rep(s2)], 1))
        d1 = self.dec1(torch.cat([self._pad(self.up1(d2), rep(s1)), rep(s1)], 1))
        d0 = self.dec0(torch.cat([self._pad(self.up0(d1), rep(s0)), rep(s0)], 1))
        return last_log.unsqueeze(1) + self.head(d0).reshape(B, To, H, W)


class ResGRUNoRec(ResGRUUNet):
    """ResGRU-UNet without recurrence: each lead hour = 3 ConvBlocks(bottleneck + learned lead embedding) (H4).
    3 blocks keep the parameter count within 0.1% of ResGRU-UNet, so capacity is not the change."""
    def __init__(self, c_in, gru_hidden=256):
        super().__init__(c_in, gru_hidden=gru_hidden)
        del self.convgru
        self.lead_emb = nn.Parameter(torch.zeros(self.t_out, gru_hidden, 1, 1))
        self.lead_block = nn.Sequential(*[ConvBlock(gru_hidden, gru_hidden, 0.1) for _ in range(3)])

    def forward(self, x, last_log):
        B, Ch, T, H, W = x.shape
        x = x.reshape(B, Ch * T, H, W)
        e0 = self.enc0(x)
        e1 = self.enc1(self.pool0(e0))
        e2 = self.enc2(self.pool1(e1))
        bn = self.bottleneck(self.pool2(e2))
        outputs = []
        for k in range(self.t_out):
            d = self.gru_proj(self.lead_block(bn + self.lead_emb[k]))
            d = self._crop(self.up2(d), e2)
            d = self.dec2(torch.cat([d, e2], 1))
            d = self._crop(self.up1(d), e1)
            d = self.dec1(torch.cat([d, e1], 1))
            d = self._crop(self.up0(d), e0)
            d = self.dec0(torch.cat([d, e0], 1))
            outputs.append(self.head(d).squeeze(1))
        return last_log.unsqueeze(1) + torch.stack(outputs, dim=1)


BACKBONES['afnov4_allskip'] = lambda: AFNOv4AllSkip()
BACKBONES['afnov4_pad'] = lambda: afno_variant(pad=True)
BACKBONES['afnov4_nosparse'] = lambda: afno_variant(sparsity=0.0)
BACKBONES['resgru_norec'] = lambda: ResGRUNoRec(C.C_IN)

# %% [markdown]
# ## 10. Run the variants of this notebook
#
# One after another, seed 42 each. After each variant, `summary.json` and `preds.npy`
# are renamed with the variant name. If the session stops, attach this output as an
# input and run again: finished variants are skipped, an interrupted one resumes.

# %%
VARIANTS = []


def run_variants():
    global BACKBONE
    for bb in VARIANTS:
        BACKBONE, C.SEED = bb, 42
        if C.IS_KAGGLE:
            C.RESUME_DIR = find_dir('/kaggle/input', [f'{bb}_s42_phaseA_last.pt'])
        main()
        for f in ('summary.json', 'preds.npy'):
            src = os.path.join(C.WORK, f)
            if os.path.exists(src):
                base, ext = os.path.splitext(f)
                os.replace(src, os.path.join(C.WORK, f'{base}_{bb}{ext}'))
        print(f'variant {bb} finished')


if __name__ == '__main__':
    run_variants()
