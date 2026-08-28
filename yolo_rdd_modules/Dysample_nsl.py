# ============================== Original Dysample.py (kept intact) ==============================
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Dy_Sample']

def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class Dy_Sample(nn.Module):
    """
    Dynamic upsample with two styles:
      - 'lp': predict offsets at low-res, then pixel shuffle
      - 'pl': pixel shuffle first, predict at high-res (requires Cin % scale^2 == 0)
    groups: split channels into groups for sampling; dyscope: learnable scope (gating) for offsets
    """
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        # (2, s, s) -> repeat for groups -> (1, 2*s*s*groups, 1, 1)
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)  # (B,2,G*s*s,H,W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1  # [-1,1]
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(
            x.reshape(B * self.groups, -1, H, W), coords,
            mode='bilinear', align_corners=False, padding_mode="border"
        ).view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)

if __name__ == '__main__':
    x = torch.rand(2, 64, 4, 7)
    dys = Dy_Sample(64)
    print("Dy_Sample:", dys(x).shape)

# ============================== Appended wrappers & registry (non-intrusive) ==============================
# 说明：
# - 不改动上面的 Dy_Sample；下面只是“封装类”，便于在 YAML 里直接写 [2, "up"]、["up", 2]、["up"] 等。
# - 采用惰性构建：第一次 forward(x) 才用 x.shape[1] 决定真实 Cin 并创建内部层，彻底避免把 2 当 c 的问题。

# --------- small utilities ---------
def _normalize_scale_mode(scale=None, mode=None, default_scale=2, default_mode='up'):
    """
    兼容 YAML 传参（顺序/类型鲁棒）：
      [2, "up"], ["up", 2], ["up"], [], ["2","up"] 等
    返回: (int_scale, mode_str)
    """
    # 尝试从 scale 得到 int
    try:
        s = int(scale)
        return s, (mode if isinstance(mode, str) else default_mode)
    except (TypeError, ValueError):
        pass
    # 尝试从 mode 得到 int（顺序被调换）
    try:
        s = int(mode)
        return s, (scale if isinstance(scale, str) else default_mode)
    except (TypeError, ValueError):
        pass
    # 两个都不是数字：回退默认
    m = scale if isinstance(scale, str) else (mode if isinstance(mode, str) else default_mode)
    return int(default_scale), m

def _ceil_to_multiple(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m

class _ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0):
        super().__init__()
        self.cv = nn.Sequential(
            nn.Conv2d(c1, c2, k, s, p, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )
    def forward(self, x): return self.cv(x)

class _DWBlur(nn.Module):
    """轻量深度卷积预平滑，抑制上采样混叠"""
    def __init__(self, c, k=3):
        super().__init__()
        self.conv = nn.Conv2d(c, c, k, 1, k//2, groups=c, bias=False)
        self.bn   = nn.BatchNorm2d(c)
        self.act  = nn.SiLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class _PWRefine(nn.Module):
    """1×1 通道细化"""
    def __init__(self, c):
        super().__init__()
        self.pw  = nn.Conv2d(c, c, 1, 1, 0, bias=False)
        self.bn  = nn.BatchNorm2d(c)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.pw(x)))

class _SE(nn.Module):
    """Squeeze-and-Excitation 注意力（大型版使用）"""
    def __init__(self, c, r=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Sequential(
            nn.Conv2d(c, max(8, c // r), 1, bias=True),
            nn.SiLU(True),
            nn.Conv2d(max(8, c // r), c, 1, bias=True),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.fc(self.avg(x))

# ---------------- 封装 1：DySampleN（等价默认 Dy_Sample） ----------------
class DySampleN(nn.Module):
    """
    与 Dy_Sample 默认行为等价：
      style='lp', groups=4, dyscope=False
    构造签名与 YAML 兼容：DySampleN(scale=2, mode='up')；支持 [2, "up"] / ["up", 2] / ["up"] / []。
    """
    def __init__(self, scale=2, mode='up', **kwargs):
        super().__init__()
        self.scale, self.mode = _normalize_scale_mode(scale, mode)
        self.built = False
        self.dy = None

    def _build(self, cin: int):
        self.dy = Dy_Sample(cin, scale=self.scale, style='lp', groups=4, dyscope=False)
        self.built = True

    def forward(self, x):
        if not self.built or (self.dy is None):
            self._build(x.shape[1])
        return self.dy(x)

# ---------------- 封装 2：DySampleS（进阶：pl + scope + 预/后处理，通道对齐到 8 的倍数） ----------------
class DySampleS(nn.Module):
    """
    更稳/更细腻：
      pre: 深度卷积预平滑
      core: Dy_Sample(style='pl', groups=8, dyscope=True)
      post: 1×1 通道细化
    注意：'pl' 需要 Cin % scale^2 == 0；本封装会把通道对齐到 8 的倍数以满足 'pl' 与 groups 的断言。
    """
    def __init__(self, scale=2, mode='up', **kwargs):
        super().__init__()
        self.scale, self.mode = _normalize_scale_mode(scale, mode)
        self.built = False
        self.align = None
        self.pre = None
        self.dy = None
        self.post = None
        self.c_aligned = None

    def _build(self, cin: int):
        multiple = 8  # LCM(4, 8)
        c_aligned = _ceil_to_multiple(cin, multiple)
        self.c_aligned = c_aligned
        self.align = nn.Identity() if c_aligned == cin else _ConvBNAct(cin, c_aligned, 1, 1, 0)
        self.pre   = _DWBlur(c_aligned, k=3)
        self.dy    = Dy_Sample(c_aligned, scale=self.scale, style='pl', groups=8, dyscope=True)
        self.post  = _PWRefine(c_aligned)
        self.built = True

    def forward(self, x):
        if not self.built:
            self._build(x.shape[1])
        x = self.align(x)
        x = self.pre(x)
        x = self.dy(x)
        x = self.post(x)
        return x

# ---------------- 封装 3：DySampleL（大型：pl→(SE)→lp，两级级联，通道对齐到 12 的倍数） ----------------
# === REPLACE ONLY THIS CLASS ===
class DySampleL(nn.Module):
    """
    大型两段式：pl(×scale) → SE → lp(×1)
    - 仅第1段 pl 做 ×scale 上采样（通常 ×2）
    - 第2段 lp 用 scale=1 做同分辨率细化（不再放大），净倍率=×2
    - 内部通道对齐到 12 的倍数以满足 'pl' 与 groups 的断言
    - 重要：输出通道 == 输入通道（末尾投影回 Cin）
    """
    def __init__(self, scale=2, mode='up', **kwargs):
        super().__init__()
        self.scale, self.mode = _normalize_scale_mode(scale, mode)
        self.built = False
        # lazy 构建成员
        self.align_in = None    # Cin -> c_aligned
        self.pre = None
        self.dy1 = None
        self.se = None
        self.dy2 = None
        self.post = None
        self.align_out = None   # c_aligned -> Cin
        self.c_aligned = None
        self.cin = None

    def _build(self, cin: int):
        self.cin = cin
        multiple = 12  # LCM(4, 12)
        c_aligned = _ceil_to_multiple(cin, multiple)
        self.c_aligned = c_aligned

        # 若已满足倍数，直接 Identity；否则用 1×1 做通道对齐
        self.align_in  = nn.Identity() if c_aligned == cin else _ConvBNAct(cin, c_aligned, 1, 1, 0)
        self.pre       = _DWBlur(c_aligned, k=5)

        # 只在第1段做上采样 ×scale
        self.dy1       = Dy_Sample(c_aligned, scale=self.scale, style='pl', groups=12, dyscope=True)
        self.se        = _SE(c_aligned, r=16)

        # 第2段同分辨率细化（scale=1）
        self.dy2       = Dy_Sample(c_aligned, scale=1,        style='lp', groups=12, dyscope=True)
        self.post      = _PWRefine(c_aligned)

        # 把内部对齐过的通道数投影回原 Cin，保证下游按 YAML 的 Cin 接口
        self.align_out = nn.Identity() if c_aligned == cin else _ConvBNAct(c_aligned, cin, 1, 1, 0)
        self.built     = True

    def forward(self, x):
        if not self.built:
            self._build(x.shape[1])  # 惰性用真实 Cin 建图
        x = self.align_in(x)   # Cin -> c_aligned
        x = self.pre(x)
        x = self.dy1(x)        # 上采样 ×scale（通常 ×2）
        x = self.se(x)
        x = self.dy2(x)        # 同分辨率细化（×1）
        x = self.post(x)
        x = self.align_out(x)  # c_aligned -> Cin（关键：输出通道与输入一致）
        return x



# ---------------- 注册表 & 工厂（可选，用于自由缝合） ----------------
UPSAMPLE_REGISTRY = {
    "DySample":   Dy_Sample,   # 原始实现
    "DySampleN":  DySampleN,   # 等价默认
    "DySampleS":  DySampleS,   # 进阶
    "DySampleL":  DySampleL,   # 大型
}

def make_upsample(name: str, *args, **kwargs):
    """
    自由缝合入口：
      name ∈ {"DySample","DySampleN","DySampleS","DySampleL"}
      - 若 name == "DySample"，kwargs 可透传 style/groups/dyscope 等自定义参数
      - 其余封装类参数固定；构造仅接受 (scale, mode)，Cin 由 forward(x) 自动探测
    """
    cls = UPSAMPLE_REGISTRY[name]
    return cls(*args, **kwargs)

# 扩展导出名（不覆盖原有 __all__）
try:
    __all__.extend(["DySampleN", "DySampleS", "DySampleL", "UPSAMPLE_REGISTRY", "make_upsample"])
except Exception:
    pass
