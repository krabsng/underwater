import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import os
import math
from .base_model import BaseModel
from timm.models.layers import trunc_normal_
from models.blocks import CBlock_ln, SwinTransformerBlock
from models.global_net import Global_pred
from utils.image_pool import ImagePool
from torchvision.models import vgg16
from loss.network_loss import LossNetwork
from models.prompt_ir_model import Downsample, Upsample
from models.networks import init_net


class Local_pred(nn.Module):
    def __init__(self, dim=16, number=4, type='ccc'):
        super(Local_pred, self).__init__()
        # initial convolution
        self.conv1 = nn.Conv2d(3, dim, 3, padding=1, groups=1)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # main blocks
        block = CBlock_ln(dim)
        block_t = SwinTransformerBlock(dim)  # head number
        if type == 'ccc':
            # blocks1, blocks2 = [block for _ in range(number)], [block for _ in range(number)]
            blocks1 = [CBlock_ln(16, drop_path=0.01), CBlock_ln(16, drop_path=0.05), CBlock_ln(16, drop_path=0.1)]
            blocks2 = [CBlock_ln(16, drop_path=0.01), CBlock_ln(16, drop_path=0.05), CBlock_ln(16, drop_path=0.1)]
        elif type == 'ttt':
            blocks1, blocks2 = [block_t for _ in range(number)], [block_t for _ in range(number)]
        elif type == 'cct':
            blocks1, blocks2 = [block, block, block_t], [block, block, block_t]
        #    block1 = [CBlock_ln(16), nn.Conv2d(16,24,3,1,1)]
        self.mul_blocks = nn.Sequential(*blocks1, nn.Conv2d(dim, 3, 3, 1, 1), nn.ReLU())
        self.add_blocks = nn.Sequential(*blocks2, nn.Conv2d(dim, 3, 3, 1, 1), nn.Tanh())

    def forward(self, img):
        img1 = self.relu(self.conv1(img))
        mul = self.mul_blocks(img1)
        add = self.add_blocks(img1)

        return mul, add


# Short Cut Connection on Final Layer
class Local_pred_S(nn.Module):
    def __init__(self, in_dim=3, dim=16, number=4, type='ccc'):
        super(Local_pred_S, self).__init__()
        # initial convolution
        self.conv1 = nn.Conv2d(in_dim, dim, 3, padding=1, groups=1)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # main blocks
        block = CBlock_ln(dim)
        block_t = SwinTransformerBlock(dim)  # head number
        if type == 'ccc':
            # blocks1 = [CBlock_ln(16, drop_path=0.01), CBlock_ln(16, drop_path=0.05), CBlock_ln(16, drop_path=0.1)]
            # blocks2 = [CBlock_ln(16, drop_path=0.01), CBlock_ln(16, drop_path=0.05), CBlock_ln(16, drop_path=0.1)]
            blocks1 = [SwinTransformerBlock(16, window_size=4, drop_path=0.01, shift_size=1),
                       SwinTransformerBlock(16, window_size=4, drop_path=0.05, shift_size=1),
                       SwinTransformerBlock(16, window_size=4, drop_path=0.01, shift_size=1)]
            blocks2 = [CBlock_ln(16, drop_path=0.01),
                       CBlock_ln(16, drop_path=0.05),
                       CBlock_ln(16, drop_path=0.1),
                       SwinTransformerBlock(16, window_size=4, drop_path=0.01, shift_size=1),
                       SwinTransformerBlock(16, window_size=4, drop_path=0.05, shift_size=1),
                       SwinTransformerBlock(16, window_size=4, drop_path=0.01, shift_size=1)]
        elif type == 'ttt':
            blocks1, blocks2 = [block_t for _ in range(number)], [block_t for _ in range(number)]
        elif type == 'cct':
            blocks1, blocks2 = [block, block, block_t], [block, block, block_t]
        #    block1 = [CBlock_ln(16), nn.Conv2d(16,24,3,1,1)]
        self.mul_blocks = nn.Sequential(*blocks1)
        self.add_blocks = nn.Sequential(*blocks2)

        self.mul_end = nn.Sequential(nn.Conv2d(dim, 3, 3, 1, 1), nn.ReLU())
        self.add_end = nn.Sequential(nn.Conv2d(dim, 3, 3, 1, 1), nn.Tanh())
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, img):
        img1 = self.relu(self.conv1(img))
        # short cut connection
        mul = self.mul_blocks(img1) + img1
        add = self.add_blocks(img1) + img1
        mul = self.mul_end(mul)
        add = self.add_end(add)

        return mul, add


class IAT(nn.Module):
    def __init__(self, in_dim=3, with_global=True, type='lol'):
        super(IAT, self).__init__()
        self.local_net = Local_pred_S(in_dim=in_dim)

        self.with_global = with_global
        if self.with_global:
            self.global_net = Global_pred(in_channels=in_dim, type=type)

    def forward(self, img_low):
        """
        # print(self.with_global)
        mul, add = self.local_net(img_low)
        img_high = (img_low.mul(mul)).add(add)

        if not self.with_global:
            # return mul, add, img_high
            return img_high
        else:
            # gamma, color = self.global_net(img_low)
            # color的shape为（8，3，3） gamma的shape是（8,1）
            gamma, color = self.global_net(img_high)
            b = img_high.shape[0]
            img_high = img_high.permute(0, 2, 3, 1)  # (B,C,H,W) -- (B,H,W,C)
            img_high = torch.stack(
                [self.apply_color(img_high[i, :, :, :], color[i, :, :]) ** gamma[i, :] for i in range(b)], dim=0)
            img_high = img_high.permute(0, 3, 1, 2)  # (B,H,W,C) -- (B,C,H,W)
            return mul, add, img_high
        """

        mul, add = self.local_net(img_low)
        img_high = (img_low.mul(mul)).add(add)
        return img_high

    def apply_color(self, image, ccm):
        # ccm是颜色矫正矩阵，shape为（8，3，3）
        shape = image.shape
        image = image.view(-1, 3)
        image = torch.tensordot(image, ccm, dims=[[-1], [-1]])
        image = image.view(shape)
        # 对张量中的元素进行截断，使其值域为（参数二，参数三）
        return torch.clamp(image, 1e-8, 1.0)


class IATModel(BaseModel):
    """
      此类使用了IAT模型，用于在没有配对数据的情况下学习图像到图像的转换。
       模型训练需要“ --dataset_mode unaligned”数据集。
       """

    def __init__(self, opt):
        super(IATModel, self).__init__(opt)
        # -----------------------krabs添加的代码--------开始-------------------- #
        # 损失的名称
        self.loss_names = ['M']
        # 定义网络,并把网络放入gpu上训练,网络命名时要以net开头，便于保存网络模型
        # self.netIAT1 = torch.nn.DataParallel(IAT(with_global=False), opt.gpu_ids)
        self.netIAT = torch.nn.DataParallel(nn.Sequential(IAT(with_global=False)), opt.gpu_ids)
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
        self.model_names = ['IAT']
        # 保存模型
        if self.isTrain:
            # 定义模型生成的图像的缓冲区，以存储以前生成的图像
            self.generateImg_pool = ImagePool(opt.pool_size)
            # 定义优化器,调整学习率的scheduler由basemodel里的函数创建
            self.optimizer = torch.optim.Adam(self.netIAT.parameters(), opt.lr, weight_decay=opt.weight_decay)
            self.optimizers.append(self.optimizer)
        # -----------------------krabs添加的代码--------结束-------------------- #

    def set_input(self, input):
        """从数据加载器解压缩输入数据并执行必要的预处理步骤.

        Parameters:
            input (dict): 包括数据本身及其元数据信息。
        """
        self.Origin_Img = input['A'].to(self.device)  # 图片为处理过后的张量
        if self.isTrain is not None:
            self.GT_Img = input['B'].to(self.device)
        self.image_paths = input['A_paths']

    def forward(self):
        # self.mul, self.add, self.Generate_Img = self.netIAT(self.Origin_Img)
        self.Generate_Img = self.netIAT(self.Origin_Img)

    def backward(self):
        # 损失函数初始权重比：1:0.04 -> 1:0.1
        self.loss_M = self.L1_smooth_loss(self.Generate_Img, self.GT_Img) + 0.05 * self.network_loss(self.Generate_Img,
                                                                                                     self.GT_Img)
        self.loss_M.backward()

    def optimize_parameters(self):
        """计算损失、梯度并更新网络权重;在每次训练迭代中调用"""
        # forward
        self.forward()  # 生成脱水图像.
        # G_A 和 G_B
        # self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds 在优化时， Gs 不需要梯度

        self.optimizer.zero_grad()  # 将IAT网络的梯度设置为零
        self.backward()  # 计算IAT网络的梯度
        self.optimizer.step()  # 更新IAT网络的权重

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """添加新的特定于数据集的选项，并重写现有选项的默认值。

        参数：
            parser          -- 原始选项解析器
            is_train (bool) -- 无论是训练阶段还是测试阶段。可以使用此标志添加特定于训练或特定于测试的选项。

        返回值:
            修改后的解析器.

        对于 CycleGAN，除了 GAN 损失外，我们还针对以下损失引入了 lambda_A、lambda_B 和 lambda_identity。
        A (源域), B (目标域).
        生成器: G_A: A -> B; G_B: B -> A.
        判别器: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (第 5.2 节 论文中的“从绘画中生成照片”)
        原始 CycleGAN 论文中未使用 Dropout。
        """
        parser.set_defaults(no_dropout=True)  # 默认 CycleGAN 不使用 dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0,  # 这里将idt损失置零
                                help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. 例如，如果身份损失的权重应小于重建损失的权重的 10 倍，请将 lambda_identity = 0.1')
            parser.add_argument('--lambda_ssim', type=float, default=10)

        return parser


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    img = torch.Tensor(1, 3, 400, 600)
    net = IAT()
    print('total parameters:', sum(param.numel() for param in net.parameters()))
    _, _, high = net(img)
