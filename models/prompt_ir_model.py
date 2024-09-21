## PromptIR: Prompting for All-in-One Blind Image Restoration
## Vaishnav Potlapalli, Syed Waqas Zamir, Salman Khan, and Fahad Shahbaz Khan
## https://arxiv.org/abs/2306.13090


import torch
# print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from models.base_model import BaseModel
from einops import rearrange
from utils.image_pool import ImagePool
from torchvision.models import vgg16
from loss.network_loss import LossNetwork
from einops.layers.torch import Rearrange
import time


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


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


class resblock(nn.Module):
    def __init__(self, dim):
        super(resblock, self).__init__()
        # self.norm = LayerNorm(dim, LayerNorm_type='BiasFree')

        self.body = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PReLU(),
                                  nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        res = self.body((x))
        res += x
        return res


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


##########################################################################
## Transformer Block
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


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
##---------- Prompt Gen Module -----------------------
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
        #(b, 384)->(b, prompt_len=5)
        prompt_weights = F.softmax(self.linear_layer(emb), dim=1)
        # unsqueeze(0)表示在第0维上新增一个纬度，squeeze(1)表示移除第1纬度
        # prompt的shape为(B, 5, 320, 16, 16)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param.unsqueeze(0).repeat(B, 1,
                                                                                                                  1, 1,
                                                                                                                  1,
                                                                                                                  1).squeeze(
            1)
        # 在指定纬度上求和操作，执行后该纬度不再存在
        #（B，5，320，16，16）-》（B，320，16，16）
        prompt = torch.sum(prompt, dim=1)
        # 通过双线性插值的方法将输入张量的空间尺寸调整为指定大小（B，320，16，16) -> （B，320，32，32)
        prompt = F.interpolate(prompt, (H, W), mode="bilinear")
        # （B，320，32，32)->（B，320，32，32)
        prompt = self.conv3x3(prompt)

        return prompt


##########################################################################
##---------- PromptIR -----------------------

class PromptIR(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 decoder=False,
                 ):

        super(PromptIR, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.decoder = decoder

        if self.decoder:
            self.prompt1 = PromptGenBlock(prompt_dim=64, prompt_len=5, prompt_size=64, lin_dim=96)
            self.prompt2 = PromptGenBlock(prompt_dim=128, prompt_len=5, prompt_size=32, lin_dim=192)
            self.prompt3 = PromptGenBlock(prompt_dim=320, prompt_len=5, prompt_size=16, lin_dim=384)

        self.chnl_reduce1 = nn.Conv2d(64, 64, kernel_size=1, bias=bias)
        self.chnl_reduce2 = nn.Conv2d(128, 128, kernel_size=1, bias=bias)
        self.chnl_reduce3 = nn.Conv2d(320, 256, kernel_size=1, bias=bias)

        self.reduce_noise_channel_1 = nn.Conv2d(dim + 64, dim, kernel_size=1, bias=bias)
        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2

        self.reduce_noise_channel_2 = nn.Conv2d(int(dim * 2 ** 1) + 128, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3

        self.reduce_noise_channel_3 = nn.Conv2d(int(dim * 2 ** 2) + 256, int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        # self.up4_3 = Upsample(int(dim * 2 ** 2))  ## From Level 4 to Level 3
        self.up4_3 = Upsample(int(dim * 2 ** 3))
        # self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 1) + 192, int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 2) + 192, int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.noise_level3 = TransformerBlock(dim=int(dim * 2 ** 2) + 512, num_heads=heads[2],
                                             ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                             LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level3 = nn.Conv2d(int(dim * 2 ** 2) + 512, int(dim * 2 ** 2), kernel_size=1, bias=bias)

        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.noise_level2 = TransformerBlock(dim=int(dim * 2 ** 1) + 224, num_heads=heads[2],
                                             ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                             LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level2 = nn.Conv2d(int(dim * 2 ** 1) + 224, int(dim * 2 ** 2), kernel_size=1, bias=bias)

        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.noise_level1 = TransformerBlock(dim=int(dim * 2 ** 1) + 64, num_heads=heads[2],
                                             ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                             LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level1 = nn.Conv2d(int(dim * 2 ** 1) + 64, int(dim * 2 ** 1), kernel_size=1, bias=bias)

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img, noise_emb=None):
        # 普通的卷机操作 (3,256,256)->(48,256,256)
        inp_enc_level1 = self.patch_embed(inp_img)
        # 4层taransformerBlock (48,256,256)->(48,256,256)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        # 下采样操作 (48,256,256)->(96,128,128)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        # 6层taransformerBlock (96,128,128)->(96,128,128)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        # (96, 128, 128)->(192, 64, 64)
        inp_enc_level3 = self.down2_3(out_enc_level2)
        # 6层taransformerBlock(192, 64, 64)->(192, 64, 64)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        # (192, 64, 64)->(384, 32, 32)
        inp_enc_level4 = self.down3_4(out_enc_level3)
        #8层taransformerBlock(384, 32, 32)->(384, 32, 32)
        latent = self.latent(inp_enc_level4)
        if self.decoder:
            """PGM块"""
            # (384, 32, 32)->(320, 32, 32)
            dec3_param = self.prompt3(latent)
            # (384, 32, 32) cat (320, 32, 32) -> (704, 32, 32)
            latent = torch.cat([latent, dec3_param], 1)
            """PIM块"""
            # taransformerBlock(704, 32, 32) -> (704, 32, 32)
            latent = self.noise_level3(latent)
            # (704, 32, 32) -> (384, 32, 32)
            latent = self.reduce_noise_level3(latent)
        # (384, 32, 32)->(192, 64, 64)
        inp_dec_level3 = self.up4_3(latent)
        ## (192, 64, 64)->(384, 64, 64)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        # (384, 64, 64)->(192, 64, 64)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        # (192, 64, 64)->(192, 64, 64)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        if self.decoder:
            dec2_param = self.prompt2(out_dec_level3)
            out_dec_level3 = torch.cat([out_dec_level3, dec2_param], 1)
            out_dec_level3 = self.noise_level2(out_dec_level3)
            out_dec_level3 = self.reduce_noise_level2(out_dec_level3)

        # (192, 64, 64)->(192, 128, 128)
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)

        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        if self.decoder:
            dec1_param = self.prompt1(out_dec_level2)
            out_dec_level2 = torch.cat([out_dec_level2, dec1_param], 1)
            out_dec_level2 = self.noise_level1(out_dec_level2)
            out_dec_level2 = self.reduce_noise_level1(out_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)

        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        # out_dec_level1 = self.output(out_dec_level1) + inp_img
        out_dec_level1 = self.output(out_dec_level1)
        return out_dec_level1

class PromptIRModel(BaseModel):
    """
      此类使用了IAT模型，用于在没有配对数据的情况下学习图像到图像的转换。
       模型训练需要“ --dataset_mode unaligned”数据集。
       """

    def __init__(self, opt):
        super(PromptIRModel, self).__init__(opt)
        # -----------------------krabs添加的代码--------开始-------------------- #
        # 损失的名称
        self.loss_names = ['M']
        # 定义网络,并把网络放入gpu上训练,网络命名时要以net开头，便于保存网络模型
        self.netPromptIR = torch.nn.DataParallel(PromptIR(), opt.gpu_ids)
        # 指定要保存的图像，训练/测试脚本将调用 <BaseModel.get_current_visuals>
        if self.isTrain is None:
            self.visual_names = ['Origin_Img', 'Generate_Img']
        else:
            self.visual_names = ['Origin_Img', 'Generate_Img', 'GT_Img']
        # 加载网络的预训权重
        # self.IAT_net.load_state_dict(torch.load(opt.pretrain_dir))
        # 定义要用到的损失
        vgg_model = vgg16(pretrained=True).features[:16]  # 定义vgg网络，加载预训练权重，并把它放到gpu上去
        vgg_model = torch.nn.DataParallel(vgg_model)  # 使用多GPU训练
        self.L1_loss = nn.L1Loss()  # 定义L1损失
        self.L1_smooth_loss = F.smooth_l1_loss  # 定义L1smooth损失
        self.network_loss = LossNetwork(vgg_model)  # 定义vgg网络损失
        self.network_loss.eval()  # 不计算梯度
        self.model_names = ['PromptIR']
        # 保存模型
        if self.isTrain:
            # 定义模型生成的图像的缓冲区，以存储以前生成的图像
            self.generateImg_pool = ImagePool(opt.pool_size)
            # 定义优化器,调整学习率的scheduler由basemodel里的函数创建
            self.optimizer = torch.optim.Adam(self.netPromptIR.parameters(), opt.lr, weight_decay=opt.weight_decay)
            self.optimizers.append(self.optimizer)
        # -----------------------krabs添加的代码--------结束-------------------- #

    def set_input(self, input):
        """从数据加载器解压缩输入数据并执行必要的预处理步骤.

        Parameters:
            input (dict): 包括数据本身及其元数据信息。
        """
        self.Origin_Img = input['A'].to(self.device)  # 图片为处理过后的张量
        if self.isTrain is not None:
            self.GT_Img= input['B'].to(self.device)
        self.image_paths = input['A_paths']

    def forward(self):
        # self.mul, self.add, self.Generate_Img = self.netIAT(self.Origin_Img)
        self.Generate_Img = self.netPromptIR(self.Origin_Img)
    def backward(self):
        # 损失函数初始权重比：1:0.04 -> 1:0.1
        self.loss_M = self.L1_smooth_loss(self.Generate_Img, self.GT_Img) + 0.05 * self.network_loss(self.Generate_Img, self.GT_Img)
        self.loss_M.backward()

    def optimize_parameters(self):
        """计算损失、梯度并更新网络权重;在每次训练迭代中调用"""
        # forward
        self.forward()  # 生成脱水图像.
        # G_A 和 G_B
        self.optimizer.zero_grad()  # 将网络的梯度设置为零
        self.backward()  # 计算网络的梯度
        self.optimizer.step()  # 更新网络的权重

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
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0,  # 这里将idt损失置零
                                help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. 例如，如果身份损失的权重应小于重建损失的权重的 10 倍，请将 lambda_identity = 0.1')
            parser.add_argument('--lambda_ssim', type=float, default=10)

        return parser

