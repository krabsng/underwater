import torch.nn as nn
import torch
import torch.nn.functional as F
import numbers
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
from einops import rearrange
from torchvision.models import vgg16
from loss.network_loss import LossNetwork
from models.base_model import BaseModel
from loss.totalvariation_loss import TotalVariationLoss
from loss.ssim_loss import SSIM
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import models
from utils import utils
from . import networks
from torch.autograd import Variable
import numpy as np

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class WindowAttention(nn.Module):
    r""" 基于窗口的多头自注意力（W-MSA）模块，带有相对位置偏置。
        支持位移窗口和非位移窗口两种模式。
    Args:
        dim (int): 输入通道的数量。
        window_size (tuple[int]): 窗口的高度和宽度。
        num_heads (int): 注意力头的数量。
        qkv_bias (bool, optional):  如果为 True，则为查询、键、值添加可学习的偏置。默认值为 True。
        qk_scale (float | None, optional): 如果设置，则覆盖默认的 head_dim ** -0.5
        attn_drop (float, optional): 注意力权重的 dropout 比率。默认值为 0.0。
        proj_drop (float, optional): 输出的 dropout 比率。默认值为 0.0。
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # 将每个窗口展开后的特征图
        # x的shape为(win_num, win_size * win_size, C)
        B_, N, C = x.shape
        # (win_num, win_size * win_size, C)->(win_num, win_size * win_size, 3*C)
        # (win_num, win_size * win_size, 3*C)->(win_num,win_size * win_size,3,)
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DualPathUpsampling(nn.Module):
    def __init__(self, in_c, num_heads, window_size):
        super().__init__()
        self.conv3 = nn.Conv2d(in_c, in_c * 2, kernel_size=3, stride=1, padding=1, bias=True)
        self.path1 = TransformerBlock(dim=in_c * 2, num_heads=num_heads, ffn_expansion_factor=2, bias=True,
                                      LayerNorm_type='BiasFree')
        self.path2 = SwinTransformerBlock(dim=in_c * 2, num_heads=num_heads, window_size=window_size)
        self.PS = nn.PixelShuffle(2)

    def forward(self, x):
        x = self.conv3(x)
        x1 = self.path1(x)
        x2 = self.path2(x)
        return self.PS(x1 + x2)


class DualPathDownsampling(nn.Module):
    def __init__(self, in_c, num_heads, window_size):
        super().__init__()
        self.conv3 = nn.Conv2d(in_c, in_c // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.path1 = TransformerBlock(dim=in_c // 2, num_heads=num_heads, ffn_expansion_factor=2, bias=True,
                                      LayerNorm_type='BiasFree')
        self.path2 = SwinTransformerBlock(dim=in_c // 2, num_heads=num_heads, window_size=window_size)
        self.PUS = nn.PixelUnshuffle(2)

    def forward(self, x):
        x = self.conv3(x)
        x1 = self.path1(x)
        x2 = self.path2(x)
        return self.PUS(x1 + x2)


class PromptGenBlock(nn.Module):
    """
        parameters:
            prompt_dim:输出的提示图的纬度
            prompt_len:序列长度
            prompt_size:提示特征图的大小
            lin_dim:输入特征图的纬度
    """

    def __init__(self, prompt_dim=128, prompt_len=5, prompt_size=96, lin_dim=192):
        super(PromptGenBlock, self).__init__()
        self.prompt_param = nn.Parameter(torch.rand(1, prompt_len, prompt_dim, prompt_size, prompt_size))
        self.linear_layer = nn.Linear(lin_dim, prompt_len)
        # 3x3的卷积不改变原始特征图的大小
        self.conv3x3 = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        # (b, 384, 32, 32)
        B, C, H, W = x.shape
        # 在最后两个纬度上求均值操作(b, 384, 32, 32)->(b, 384)
        emb = x.mean(dim=(-2, -1))
        # (b, 384)->(b, prompt_len=5)
        prompt_weights = F.softmax(self.linear_layer(emb), dim=1)
        # unsqueeze(0)表示在第0维上新增一个纬度，squeeze(1)表示移除第1纬度
        # prompt的shape为(B, 5, 320, 16, 16)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param.unsqueeze(0).repeat(B, 1,
                                                                                                                  1, 1,
                                                                                                                  1,
                                                                                                                  1).squeeze(
            1)
        # 在指定纬度上求和操作，执行后该纬度不再存在
        # （B，5，320，16，16）-》（B，320，16，16）
        prompt = torch.sum(prompt, dim=1)
        # 通过双线性插值的方法将输入张量的空间尺寸调整为指定大小（B，320，16，16) -> （B，320，32，32)
        prompt = F.interpolate(prompt, (H, W), mode="bilinear")
        # （B，320，32，32)->（B，320，32，32)
        prompt = self.conv3x3(prompt)

        return prompt


class Downsample(nn.Module):
    """
        下采样，特征图大小变为原来的一半，纬度变为原来的两倍
    """

    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    """
        上采样，特征图大小变为原来的两倍，纬度变为原来的一半
    """

    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


def window_reverse(windows, window_size, H, W):
    """
    fucntion：
        这个函数的作用是将经过 window_partition 函数分割后的多个小窗口，
        按照原始图像的形状重新拼接起来，使得这些窗口组合回原来的图像结构。
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Mlp(nn.Module):
    """
        一个简单的多层感知机（MLP）模块,通过两个全连接层（线性层）对输入数据进行非线性变换
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    function：输入的图像张量 x 分割成大小为 window_size x window_size 的小窗口，并重新排列这些窗口，以便后续处理。
    Args:
        x: (B, H, W, C)
        window_size (int): 窗口大小
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    # x的shape为(B,H,W,C)
    B, H, W, C = x.shape
    # 将张量划分为若干个不同的窗口(B,H,W,C)->(B,H/win_size,win_size,W/win_size,win_size,C)
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # (B,H/win_size,win_size,W/win_size,win_size,C) -> (B,H/win_size,W/win_size,win_size,win_size,C)->(win_num,win_size,win_size,C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


class Aff_channel(nn.Module):
    """
        function：
            Aff_channel 模块的主要功能是在不同的通道排列（即 channel_first 或 channel_last）下，
            对输入张量的每个通道进行线性缩放、偏移和颜色调整。它通过学习参数 alpha、beta 和 color 来对输入数据进行调整。
        不该变输入特征图的尺寸
    """

    def __init__(self, dim, channel_first=True):
        super().__init__()
        # learnable
        self.alpha = nn.Parameter(torch.ones([1, 1, dim]))
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]))
        self.color = nn.Parameter(torch.eye(dim))
        self.channel_first = channel_first

    def forward(self, x):
        # x的shape(B,H*W,C)
        if self.channel_first:
            # 在指定维度上执行求和的操作
            x1 = torch.tensordot(x, self.color, dims=[[-1], [-1]])
            x2 = x1 * self.alpha + self.beta
        else:
            x1 = x * self.alpha + self.beta
            x2 = torch.tensordot(x1, self.color, dims=[[-1], [-1]])
        return x2


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): 输入通道的数量。
        input_resolution (tuple[int]): 输入的分辨率。
        num_heads (int): 注意力头的数量
        window_size (int): 窗口大小。
        shift_size (int): 用于 SW-MSA 的位移大小。
        mlp_ratio (float): MLP 隐藏层维度与嵌入维度的比例。
        qkv_bias (bool, optional): 如果为 True，则为查询、键、值添加可学习的偏置。默认值：True。
        qk_scale (float | None, optional): 如果设置，则覆盖默认的 head_dim ** -0.5
        drop (float, optional): Dropout 率。默认值：0.0。
        attn_drop (float, optional): 注意力层的 dropout 率。默认值：0.0
        drop_path (float, optional): 随机深度率。默认值：0.0
        act_layer (nn.Module, optional): 激活层。默认值：nn.GELU。
        norm_layer (nn.Module, optional): 激活层。默认值：nn.GELU。
    """

    def __init__(self, dim, num_heads=2, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=Aff_channel):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        # self.norm1 = norm_layer(dim)
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.norm2 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # 深度可分离卷积
        x = x + self.pos_embed(x)
        B, C, H, W = x.shape
        # 从第二个维度将进行展平，然后交换
        # （B,C,H,W）-> (B,C,H*W)->(B,H*W,C)
        x = x.flatten(2).transpose(1, 2)

        shortcut = x
        # 通道重排列
        # (B,H*W,C)->(B,H*W,C)
        x = self.norm1(x)
        # (B,H*W,C)->(B,H,W,C)
        x = x.view(B, H, W, C)

        # 循环移位
        if self.shift_size > 0:
            # (B, H, W, C) -> (B,H,W,C)
            # torch.roll函数的作用是将张量x的元素沿指定的维度进行循环移位, （←，↑）
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # 划分窗口窗口
        # (B,H,W,C)->(win_num,win_size,win_size,C)
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        # (win_num,win_size,win_size,C)->(win_num,win_size*win_size,C)，将每个窗口展开
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # 展开每个窗口后的特征图的shape(win_num,win_size*win_size,C)
        # W-MSA/SW-MSA 窗口多头自注意力/平移窗口多头自注意力
        attn_windows = self.attn(x_windows)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).reshape(B, C, H, W)

        return x


class OverlapPatchEmbed(nn.Module):
    """
        提取高纬度的特征
    """

    def __init__(self, in_c=3, out_c=48, bias=False, type="aa"):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=bias)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.proj(x)
        x = self.tanh(x)
        return x


# 定义生成器
class SPUNet(nn.Module):
    """
        生成器结构，采用4层编码，4层解码结构
    """

    def __init__(self, in_dim=3, mid_dim=16, out_dim=3, num_blocks=[1, 1, 1, 1], num_heads=[4, 2, 2, 1],
                 win_sizes=[4, 4, 2, 2], Prompt=False, SR=False):
        super(SPUNet, self).__init__()
        self.SR = SR
        self.Prompt = Prompt
        # 提取高纬度特征
        self.pro1 = OverlapPatchEmbed(in_c=in_dim, out_c=mid_dim)
        # 从高维返回低维
        self.unpro1 = OverlapPatchEmbed(in_c=mid_dim, out_c=out_dim)
        self.encoder1 = nn.Sequential(
            *[SwinTransformerBlock(dim=int(mid_dim * 2 ** 0), num_heads=num_heads[0], window_size=win_sizes[0])
              for i in range(num_blocks[0])]
        )
        self.down1 = Downsample(int(mid_dim * 2 ** 0))

        self.encoder2 = nn.Sequential(
            *[SwinTransformerBlock(dim=int(mid_dim * 2 ** 1), num_heads=num_heads[1], window_size=win_sizes[1])
              for i in range(num_blocks[1])]
        )
        self.down2 = Downsample(int(mid_dim * 2 ** 1))

        self.encoder3 = nn.Sequential(
            *[SwinTransformerBlock(dim=int(mid_dim * 2 ** 2), num_heads=num_heads[2], window_size=win_sizes[2])
              for i in range(num_blocks[2])]
        )
        self.down3 = Downsample(int(mid_dim * 2 ** 2))

        self.encoder4 = nn.Sequential(
            *[SwinTransformerBlock(dim=int(mid_dim * 2 ** 3), num_heads=num_heads[3], window_size=win_sizes[3])
              for i in range(num_blocks[3])]
        )

        self.down4 = Downsample(int(mid_dim * 2 ** 3))

        self.up4 = Upsample(int(mid_dim * 2 ** 4))
        self.decoder4 = nn.Sequential(
            *[SwinTransformerBlock(dim=int(mid_dim * 2 ** 3), num_heads=num_heads[3], window_size=win_sizes[3])
              for i in range(num_blocks[3])]
        )
        self.reduce_c4 = nn.Conv2d(int(mid_dim * 2 ** 4), int(mid_dim * 2 ** 3), kernel_size=1, bias=True)

        self.up3 = Upsample(int(mid_dim * 2 ** 3))
        self.decoder3 = nn.Sequential(
            *[SwinTransformerBlock(dim=int(mid_dim * 2 ** 2), num_heads=num_heads[2], window_size=win_sizes[2])
              for i in range(num_blocks[2])]
        )
        self.reduce_c3 = nn.Conv2d(int(mid_dim * 2 ** 3), int(mid_dim * 2 ** 2), kernel_size=1, bias=True)

        self.up2 = Upsample(int(mid_dim * 2 ** 2))
        self.decoder2 = nn.Sequential(
            *[SwinTransformerBlock(dim=int(mid_dim * 2 ** 1), num_heads=num_heads[1], window_size=win_sizes[1])
              for i in range(num_blocks[1])]
        )
        self.reduce_c2 = nn.Conv2d(int(mid_dim * 2 ** 2), int(mid_dim * 2 ** 1), kernel_size=1, bias=True)

        self.up1 = Upsample(int(mid_dim * 2 ** 1))
        self.decoder1 = nn.Sequential(
            *[SwinTransformerBlock(dim=int(mid_dim * 2 ** 0), num_heads=num_heads[0], window_size=win_sizes[0])
              for i in range(num_blocks[1])]
        )
        self.reduce_c1 = nn.Conv2d(int(mid_dim * 2 ** 1), int(mid_dim * 2 ** 0), kernel_size=1, bias=True)

        self.up_sr = nn.Sequential(*([DualPathUpsampling(int(mid_dim * 2 ** 0), 2, 4)] +
                                     [SwinTransformerBlock(dim=int(mid_dim // 2), num_heads=2, window_size=2)
                                      for i in range(2)]))
        self.dw_lr = nn.Sequential(*[DualPathDownsampling(int(mid_dim * 2 ** -1), 2, 4)])
        self.lr_p = OverlapPatchEmbed(in_c=mid_dim, out_c=out_dim)
        self.sr_p = OverlapPatchEmbed(in_c=mid_dim // 2, out_c=out_dim)
        if self.Prompt:
            """
                prompt_dim:输出的提示图的纬度
                prompt_len:序列长度
                prompt_size:提示特征图的大小
                lin_dim:输入特征图的纬度
            """
            self.prompt3 = PromptGenBlock(prompt_dim=int(mid_dim * 2 ** 3), prompt_len=5, prompt_size=16,
                                          lin_dim=int(mid_dim * 2 ** 3))
            self.noise3 = SwinTransformerBlock(dim=int(mid_dim * 2 ** 4), num_heads=4, window_size=2)
            self.reduce_noise3 = nn.Conv2d(int(mid_dim * 2 ** 4), int(mid_dim * 2 ** 3), kernel_size=1, bias=True)

            self.prompt2 = PromptGenBlock(prompt_dim=int(mid_dim * 2 ** 2), prompt_len=5, prompt_size=32,
                                          lin_dim=int(mid_dim * 2 ** 2))
            self.noise2 = SwinTransformerBlock(dim=int(mid_dim * 2 ** 3), num_heads=4, window_size=2)
            self.reduce_noise2 = nn.Conv2d(int(mid_dim * 2 ** 3), int(mid_dim * 2 ** 2), kernel_size=1, bias=True)

            self.prompt1 = PromptGenBlock(prompt_dim=int(mid_dim * 2 ** 1), prompt_len=5, prompt_size=32,
                                          lin_dim=int(mid_dim * 2 ** 1))
            self.noise1 = SwinTransformerBlock(dim=int(mid_dim * 2 ** 2), num_heads=4, window_size=2)
            self.reduce_noise1 = nn.Conv2d(int(mid_dim * 2 ** 2), int(mid_dim * 2 ** 1), kernel_size=1, bias=True)

    def forward(self, x):
        # (3, 256, 256) -> (32, 256, 256)
        x = self.pro1(x)
        e1 = self.encoder1(x)
        # (32, 256, 256) -> (64, 128, 128)
        e_d1 = self.down1(e1)
        e2 = self.encoder2(e_d1)
        # (64, 128, 128) -> (128, 64, 64)
        e_d2 = self.down2(e2)
        e3 = self.encoder3(e_d2)
        # (128, 64, 64) -> (256, 32, 32)
        e_d3 = self.down3(e3)
        e4 = self.encoder4(e_d3)
        # (256, 32, 32) -> (512, 16, 16)
        e_d4 = self.down4(e4)

        # (512, 16, 16) -> (256, 32, 32)
        d_u4 = self.up4(e_d4)
        rc4 = self.reduce_c4(torch.cat([d_u4, e4], dim=1))  # ↓
        d4 = self.decoder4(rc4)
        if self.Prompt:
            prompt3 = self.prompt3(d4)
            prompt3 = torch.cat([d4, prompt3], dim=1)
            prompt3 = self.noise3(prompt3)
            d4 = self.reduce_noise3(prompt3)

        # (256, 32, 32) -> (128, 64, 64)
        d_u3 = self.up3(d4)
        rc3 = self.reduce_c3(torch.cat([d_u3, e3], dim=1))
        d3 = self.decoder3(rc3)
        if self.Prompt:
            prompt2 = self.prompt2(d3)
            prompt2 = torch.cat([d3, prompt2], dim=1)
            prompt2 = self.noise2(prompt2)
            d3 = self.reduce_noise2(prompt2)

        # (128, 64, 64) -> (64, 128, 128)
        d_u2 = self.up2(d3)
        rc2 = self.reduce_c2(torch.cat([d_u2, e2], dim=1))
        d2 = self.decoder2(rc2)
        # (64, 128, 128) -> (32, 256, 256)
        if self.Prompt:
            prompt1 = self.prompt1(d2)
            prompt1 = torch.cat([d2, prompt1], dim=1)
            prompt1 = self.noise1(prompt1)
            d2 = self.reduce_noise1(prompt1)

        d_u1 = self.up1(d2)
        rc1 = self.reduce_c1(torch.cat([d_u1, e1], dim=1))
        out = self.decoder1(rc1)
        # out = self.unpro1(out)

        # ----------------修改--开始----------------------- #
        # (32, 256, 256) -> (16, 512, 512)
        sr = self.up_sr(out)
        if not self.SR:
            # (16, 512, 512) -> (32, 256, 256)
            lr = self.dw_lr(sr)
            return self.lr_p(lr)
        return self.sr_p(sr)
        # ----------------修改--结束----------------------- #


class VGG19_Discriminator(nn.Module):
    """
        定义VGG19作为判别器
    """

    def __init__(self, _pretrained_=True):
        super(VGG19_Discriminator, self).__init__()
        self.vgg = models.vgg19(pretrained=_pretrained_).features
        # 冻结VGG19卷积层的参数
        for param in self.vgg.parameters():
            param.requires_grad_(False)
        # 自适应平均池化，将输出调整为 7x7（可根据需要调整）
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        # 添加判别器的自定义全连接层
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * 8 * 8, 512)  # 假设输入为224x224的图像
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()  # 输出为1表示真实，0表示生成

    def forward(self, x):
        # 使用VGG19提取特征
        x = self.vgg(x)
        x = self.adaptive_pool(x)
        # 展平并通过全连接层
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x


class SPUGANModel(BaseModel):
    """
        此类使用了自定义的Krabs模型，用于在配对数据的情况下学习图像到图像的转换。
       模型训练需要“ --dataset_mode unaligned”数据集。
       """

    def __init__(self, opt):
        super(SPUGANModel, self).__init__(opt)
        self.opt = opt
        self.SR = True  # 是否是进行超分辨率训练
        self.Prompt = True # 是否使用提示学习
        # 损失的名称
        self.loss_names = ["G", "D"]
        # 定义网络,并把网络放入gpu上训练,网络命名时要以net开头，便于保存网络模型
        self.netSPU = SPUNet(SR=False, Prompt=self.Prompt).to(self.device)
        if self.isTrain:
            self.netD = networks.define_D(6, 64, "pixel",
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        if opt.distributed:
            self.netSPU = DDP(self.netSPU, device_ids=[opt.gpu])
            # self.netD = DDP(self.netD, device_ids=[opt.gpu])
        else:
            self.netSPU = torch.nn.DataParallel(self.netSPU, opt.gpu_ids).to(self.device)
            # self.netD = torch.nn.DataParallel(self.netD, opt.gpu_ids).to(self.device)
        # self.netKrabs = torch.nn.DataParallel(KrabsNet(SR=self.SR), opt.gpu_ids)
        # self.netKrabs = self.netKrabs.to(self.device)

        # 指定要保存的图像，训练/测试脚本将调用 <BaseModel.get_current_visuals>
        if self.isTrain is None:
            self.visual_names = ['Origin_Img', 'Generate_Img']
        else:
            self.visual_names = ['Origin_Img', 'Generate_Img', 'GT_Img']
        if not self.SR:
            # 加载网络的预训权重
            if isinstance(self.netSPU, torch.nn.DataParallel):
                self.netSPU.module.load_state_dict(
                    torch.load('/a.krabs/krabs/checkpoints/krabs_net_sr/100_net_Krabs.pth'))
            else:
                self.netSPU.load_state_dict(
                    torch.load('/a.krabs/krabs/checkpoints/krabs_net_sr/100_net_Krabs.pth'))

        #region 定义一些要用到的损失函数
        vgg_model = vgg16(pretrained=True).features[:16]  # 定义vgg网络，加载预训练权重，并把它放到gpu上去
        vgg_model = vgg_model.to(self.device)
        self.L1_loss = nn.MSELoss()  # 定义L1损失
        self.ssim_loss = SSIM()  # 定义L1smooth损失
        self.network_loss = LossNetwork(vgg_model)  # 定义vgg网络损失
        self.TotalVariation_loss = TotalVariationLoss()
        self.network_loss.eval()  # 不计算梯度
        self.criterionGAN = networks.GANLoss("lsgan").to(self.device)  # 定义GAN损失.
        #endregion

        self.model_names = ['SPU']
        # 保存模型
        if self.isTrain:
            # 定义优化器,调整学习率的scheduler由basemodel里的函数创建
            self.optimizer_G = torch.optim.Adam(self.netSPU.parameters(), opt.lr, weight_decay=opt.weight_decay)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), opt.lr, weight_decay=opt.weight_decay)
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """从数据加载器解压缩输入数据并执行必要的预处理步骤.

        Parameters:
            input (dict): 包括数据本身及其元数据信息。
        """
        self.Origin_Img = input['A'].to(self.device)  # 图片为处理过后的张量
        if self.isTrain is not None:
            self.GT_Img = input['B'].to(self.device)
        if self.SR:
            self.Origin_Pro_Img = F.interpolate(self.Origin_Img, scale_factor=2, mode='bicubic', align_corners=False).to(self.device) # bilinear 双线性
        self.image_paths = input['A_paths']

    def forward(self):

        # # 检查模型的第一个参数在哪个设备上
        # print(torch.cuda.device_count())  # 确保有足够数量的 GPU
        # print(torch.cuda.get_device_name(0))  # 检查每个 GPU 的名称
        #
        # print(self.opt.gpu_ids)
        # print(next(self.netKrabs.parameters()).device)
        # print(self.Origin_Img.device)
        # for param in self.netKrabs.parameters():
        #     print(param.device)

        self.Generate_Img = self.netSPU(self.Origin_Pro_Img)

    def backward_G(self):
        # 损失函数初始权重比：1:0.04 -> 1:0.1
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_C = self.opt.lambda_C
        lambda_D = self.opt.lambda_D
        lambda_E = self.opt.lambda_E

        fake_AB = torch.cat((self.Origin_Pro_Img, self.Generate_Img), 1)
        pred_fake = self.netD(fake_AB)

        self.loss_G = lambda_A * self.ssim_loss(self.Generate_Img, self.GT_Img) + \
                      lambda_B * self.network_loss(self.Generate_Img, self.GT_Img) + \
                      lambda_C * self.L1_loss(self.Generate_Img, self.GT_Img) + \
                      lambda_D * self.TotalVariation_loss(self.Generate_Img) + \
                      lambda_E * self.criterionGAN(pred_fake, True)
        self.loss_G = self.loss_G
        self.loss_G.backward()


    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.Origin_Pro_Img, self.Generate_Img),
                            1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.Origin_Pro_Img, self.GT_Img), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def optimize_parameters(self):
        """计算损失、梯度并更新网络权重;在每次训练迭代中调用"""
        # forward
        self.forward()  # 生成脱水图像.

        # 优化生成器
        self.set_requires_grad([self.netD], False)
        self.optimizer_G.zero_grad()  # 将 G的梯度设置为零
        self.backward_G()  # 计算G的梯度
        self.optimizer_G.step()  # 更新G权重
        # 优化判别器
        self.set_requires_grad([self.netD], True)
        self.optimizer_D.zero_grad()  # 将D的梯度设置为零
        self.backward_D()  # 计算D梯度
        self.optimizer_D.step()  # 更新D权重

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """添加新的特定于数据集的选项，并重写现有选项的默认值。

        参数：
            parser          -- 原始选项解析器
            is_train (bool) -- 无论是训练阶段还是测试阶段。可以使用此标志添加特定于训练或特定于测试的选项。

        返回值:
            修改后的解析器.

        """
        parser.set_defaults(no_dropout=True)  # 默认 CycleGAN 不使用 dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=0, help='')  # SSIM
            parser.add_argument('--lambda_B', type=float, default=0, help='')  # netWork
            parser.add_argument('--lambda_C', type=float, default=0, help='')  # L2
            parser.add_argument('--lambda_D', type=float, default=1, help='')  # 全变差
            parser.add_argument('--lambda_E', type=float, default=0, help='')  # GAN
        return parser
