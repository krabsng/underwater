import itertools
import torch
import torch.nn as nn
from utils.image_pool import ImagePool
from .base_model import BaseModel
from loss.network_loss import VGG19_PercepLoss
from torch.autograd import Variable
import numpy as np

from . import networks


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, bn=True):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        # 批量归一化
        if bn: layers.append(nn.BatchNorm2d(out_size, momentum=0.8))
        layers.append(nn.LeakyReLU(0.2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUp, self).__init__()
        layers = [
            # 反卷积
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_size, momentum=0.8),
            nn.ReLU(inplace=True),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x


class GeneratorFunieGAN(nn.Module):
    """ A 5-layer UNet-based generator as described in the paper
    """

    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorFunieGAN, self).__init__()
        # encoding layers
        self.down1 = UNetDown(in_channels, 32, bn=False)
        self.down2 = UNetDown(32, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 256)
        self.down5 = UNetDown(256, 256, bn=False)
        # decoding layers
        self.up1 = UNetUp(256, 256)
        self.up2 = UNetUp(512, 256)
        self.up3 = UNetUp(512, 128)
        self.up4 = UNetUp(256, 32)
        self.final = nn.Sequential(
            # 上采样
            nn.Upsample(scale_factor=2),
            # 对张量进行零填充
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(64, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # (3, 256, 256)->(32, 128, 128)
        d1 = self.down1(x)
        # (32, 128, 128)->(128, 64, 64)
        d2 = self.down2(d1)
        # (128, 64, 64)->(256, 32, 32)
        d3 = self.down3(d2)
        # (256, 32, 32)->(256, 16, 16)
        d4 = self.down4(d3)
        # (256, 32, 32)->(256, 8, 8)
        d5 = self.down5(d4)
        # (256, 8, 8)->(256, 16, 16) + (256, 16, 16) = (512, 16, 16)
        u1 = self.up1(d5, d4)
        # (512, 16, 16)->(256, 32, 32) + (256, 32, 32) = (512, 32, 32)
        u2 = self.up2(u1, d3)
        # (512, 32, 32)->(128, 64, 64) + (128, 64, 64) = (256, 64, 64)
        u3 = self.up3(u2, d2)
        # (256, 64, 64)->(32, 128, 128) + (32, 128, 128) = (64, 128, 128)
        u45 = self.up4(u3, d1)
        # (64, 128, 128) -> (3, 256, 256) 值的范围采用了Tanh激活所以值为(-1,1)
        return self.final(u45)


class DiscriminatorFunieGAN(nn.Module):
    """ A 4-layer Markovian discriminator as described in the paper
    """

    def __init__(self, in_channels=3):
        super(DiscriminatorFunieGAN, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            # Returns downsampling layers of each discriminator block
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if bn: layers.append(nn.BatchNorm2d(out_filters, momentum=0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            # (6,256,256)->(32, 128, 128)
            *discriminator_block(in_channels * 2, 32, bn=False),
            # (32, 128, 128)->(64, 64, 64)
            *discriminator_block(32, 64),
            # (64, 64, 64)->(128, 32, 32)
            *discriminator_block(64, 128),
            # (128, 32, 32)->(256, 16, 16)
            *discriminator_block(128, 256),
            # (256, 16, 16)->(256, 17, 17)
            nn.ZeroPad2d((1, 0, 1, 0)),
            # (256, 17, 17)->(1, 16, 16)
            nn.Conv2d(256, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)


class FUNIEGANModel(BaseModel):
    """
    此类实现了 FUNIEGAN 模型。
    模型训练需要“--dataset_mode underwater——dataset”数据集。
    源论文地址: https://arxiv.org/pdf/1903.09766.pdf 2019年发布
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """添加新的特定于数据集的选项，并重写现有选项的默认值。
        参数：
            parser          -- 原始选项解析器
            is_train (bool) -- 无论是训练阶段还是测试阶段。可以使用此标志添加特定于训练或特定于测试的选项。

        返回值:
            修改后的解析器.
        """
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=7, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=3, help='weight for cycle loss (B -> A -> B)')
        return parser

    def __init__(self, opt):
        """初始化 FUNIEGAN 类。

        Parameters:
            opt (Option class)-- 存储所有实验标志;需要是 BaseOptions 的子类
        """
        BaseModel.__init__(self, opt)
        # 指定要打印的训练损失. 训练/测试脚本将调用 <BaseModel.get_current_losses>
        """损失函数的构成
                判别器D的损失 loss_D = ||D(GT_Img, Origin_Img), True|| + ||(G(Origin_Img), Origin_Img), False||
                生成器G的损失 loss_G = ||(G(Origin_Img), Origin_Img), True|| + ||G(Origin_Img), GT_Img|| + VGG(G(Origin_Img), GT_Img)

            三类不同的图像
                Origin_Img:原始的图片   Generate_Img: 生成器生成的图片   GT_Img：目标图片
        """
        self.loss_names = ['D', 'G']
        # 指定要保存/显示的图像. 训练/测试脚本将调用 <BaseModel.get_current_visuals>
        self.visual_names = ['Origin_Img', 'Generate_Img', 'GT_Img']
        # 指定要保存到磁盘的模型. 训练/测试脚本将调用<BaseModel.save_networks> 和 <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # 在测试期间，仅加载 Gs
            self.model_names = ['G']

        # 定义网络（生成器和判别器）
        # 命名与论文中使用的命名不同。
        self.netG = torch.nn.DataParallel(GeneratorFunieGAN().to(self.device), opt.gpu_ids)

        if self.isTrain:  # 定义判别器
            self.netD = torch.nn.DataParallel(DiscriminatorFunieGAN().to(self.device),opt.gpu_ids)

        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)  # 创建图像缓冲区以存储以前生成的图像
            self.fake_B_pool = ImagePool(opt.pool_size)  # 创建图像缓冲区以存储以前生成的图像
            # 定义损失函数
            self.Adv_cGAN = torch.nn.MSELoss()
            self.L1_G = torch.nn.L1Loss()
            self.L_vgg = VGG19_PercepLoss()
            torch.nn.DataParallel(self.L_vgg)  # 放到GPU上去
            self.L_vgg.eval()

            # 初始化优化器;调度程序将由函数 <BaseModel.setup> 自动创建。
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """从数据加载器解压缩输入数据并执行必要的预处理步骤.

        Parameters:
            input (dict): 包括数据本身及其元数据信息。

        “direction”选项可用于交换域 A 和域 B。
        """
        AtoB = self.opt.direction == 'AtoB'
        self.Origin_Img = input['A' if AtoB else 'B'].to(self.device)
        self.GT_Img = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        # 标签真和标签假
        self.valid = Variable(torch.cuda.FloatTensor(
            np.ones((self.Origin_Img.size(0), *(1, self.Origin_Img.size(2) // 16, self.Origin_Img.size(3) // 16)))),
                              requires_grad=False)
        self.fake = Variable(torch.cuda.FloatTensor(
            np.zeros((self.Origin_Img.size(0), *(1, self.Origin_Img.size(2) // 16, self.Origin_Img.size(3) // 16)))),
                             requires_grad=False)

    def forward(self):
        """前向传播;由两个函数调用 <optimize_parameters> and <test>."""
        self.Generate_Img = self.netG(self.Origin_Img)

    def backward_D(self):
        """计算判别器D的损失"""
        # 此时判别器网络应预测为真
        pred_real = self.netD(self.GT_Img, self.Origin_Img)
        loss_real = self.Adv_cGAN(pred_real, self.valid)
        pred_fake = self.netD(self.Generate_Img, self.Origin_Img)
        loss_fake = self.Adv_cGAN(pred_fake, self.fake)
        self.loss_D = 0.5 * (loss_real + loss_fake) * 10.0  # 10x scaled for stability
        self.loss_D.backward(retain_graph=True)

    def backward_G(self):
        """计算生成器G的损失"""
        lambda_A = self.opt.lambda_A  # 损失的权重           默认值为7
        lambda_B = self.opt.lambda_B  # 损失的权重           默认值为3
        pred_fake = self.netD(self.Generate_Img, self.Origin_Img)
        loss_GAN = self.Adv_cGAN(pred_fake, self.valid)  # GAN loss
        loss_1 = self.L1_G(self.Generate_Img, self.GT_Img)  # similarity loss
        loss_con = self.L_vgg(self.Generate_Img, self.GT_Img)  # content loss
        # combined loss and calculate gradients
        self.loss_G = loss_GAN + lambda_A * loss_1 + lambda_B * loss_con
        self.loss_G.backward()

    def optimize_parameters(self):
        """计算损失、梯度并更新网络权重;在每次训练迭代中调用"""
        # forward
        self.forward()  # 计算假图像和重建图像.
        # D
        self.set_requires_grad([self.netD], True)
        self.optimizer_D.zero_grad()  # 将D的梯度设置为零
        self.backward_D()
        self.optimizer_D.step()  # 更新D权重
        # G
        self.set_requires_grad([self.netD], False)  # Ds 在优化时， Gs 不需要梯度
        self.optimizer_G.zero_grad()  # 将 G的梯度设置为零
        self.backward_G()  # 计算 G的梯度
        self.optimizer_G.step()  # 更新G权重

