import torch
import itertools
from utils.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torch.nn as nn
from torchvision.models import vgg16
from loss.network_loss import LossNetwork


class ConfidenceMapGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        # Confidence maps
        # 输入数据的尺寸 (N, 3*4, H, W)
        # 输出数据的尺寸 (N, 3, H, W)
        self.conv1 = nn.Conv2d(
            in_channels=12, out_channels=128, kernel_size=7, dilation=1, padding="same"
            # Padding = "same"（表示填充使得输出尺寸与输入尺寸相同）
        )
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=5, dilation=1, padding="same"
        )
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, dilation=1, padding="same"
        )
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=64, kernel_size=1, dilation=1, padding="same"
        )
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=7, dilation=1, padding="same"
        )
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=5, dilation=1, padding="same"
        )
        self.relu6 = nn.ReLU()
        self.conv7 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, dilation=1, padding="same"
        )
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv2d(
            in_channels=64, out_channels=3, kernel_size=3, dilation=1, padding="same"
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, wb, ce, gc):
        out = torch.cat([x, wb, ce, gc], dim=1)
        out = self.relu1(self.conv1(out))
        out = self.relu2(self.conv2(out))
        out = self.relu3(self.conv3(out))
        out = self.relu4(self.conv4(out))
        out = self.relu5(self.conv5(out))
        out = self.relu6(self.conv6(out))
        out = self.relu7(self.conv7(out))
        out = self.sigmoid(self.conv8(out))
        out1, out2, out3 = torch.split(out, [1, 1, 1], dim=1)
        return out1, out2, out3


class Refiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=6, out_channels=32, kernel_size=7, dilation=1, padding="same"
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=5, dilation=1, padding="same"
        )
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=3, kernel_size=3, dilation=1, padding="same"
        )
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

    def forward(self, x, xbar):
        out = torch.cat([x, xbar], dim=1)
        out = self.relu1(self.conv1(out))
        out = self.relu2(self.conv2(out))
        out = self.relu3(self.conv3(out))
        return out


class WaterNet(nn.Module):
    """
    waternet = WaterNet()
    in = torch.randn(16, 3, 112, 112)
    waternet_out = waternet(in, in, in, in)
    waternet_out.shape
    # torch.Size([16, 3, 112, 112])
    """

    def __init__(self):
        super().__init__()
        self.cmg = ConfidenceMapGenerator()
        self.wb_refiner = Refiner()
        self.ce_refiner = Refiner()
        self.gc_refiner = Refiner()

    """
        x:表示原始的 RGB 图像
        wb:原始图像经白平衡处理后的图像
        ce：直方图处理后的图像
        gc：伽马矫正后的图像
    """

    def forward(self, x, wb, ce, gc):
        wb_cm, ce_cm, gc_cm = self.cmg(x, wb, ce, gc)
        refined_wb = self.wb_refiner(x, wb)
        refined_ce = self.ce_refiner(x, ce)
        refined_gc = self.gc_refiner(x, gc)
        return (
                torch.mul(refined_wb, wb_cm)
                + torch.mul(refined_ce, ce_cm)
                + torch.mul(refined_gc, gc_cm)
        )


class WaterNetModel(BaseModel):
    """
   此类实现了 WaterNet 模型，

    模型训练需要“--dataset_mode waternet_dataset”数据集。
    WaterNet原论文地址：
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
        parser.set_defaults(no_dropout=True)  # 默认 CycleGAN 不使用 dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=80.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=20.0, help='weight for cycle loss (B -> A -> B)')

        return parser

    def __init__(self, opt):
        """初始化 WaterNet 类。

        Parameters:
            opt (Option class)-- 存储所有实验标志;需要是 BaseOptions 的子类
        """
        BaseModel.__init__(self, opt)
        self.netWaterNet = torch.nn.DataParallel(WaterNet(), opt.gpu_ids)
        self.loss_names = ['M']
        # 指定要保存/显示的图像. 训练/测试脚本将调用 <BaseModel.get_current_visuals>
        self.visual_names = ['origin_img', 'wb_img', 'ce_img', 'gc_img', 'generate_img', 'gt_img']
        # 指定要保存到磁盘的模型. 训练/测试脚本将调用<BaseModel.save_networks> 和 <BaseModel.load_networks>.
        self.model_names = ['WaterNet']

        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)  # 创建图像缓冲区以存储以前生成的图像
            # 定义损失函数
            vgg_model = vgg16(pretrained=True).features[:16]  # 定义vgg网络，加载预训练权重，并把它放到gpu上去
            vgg_model = torch.nn.DataParallel(vgg_model)  # 使用多GPU训练
            self.perceptual_loss = nn.MSELoss()
            self.network_loss = LossNetwork(vgg_model)  # 定义vgg网络损失

            # 初始化优化器;调度程序将由函数 <BaseModel.setup> 自动创建。
            self.optimizer = torch.optim.Adam(itertools.chain(self.netWaterNet.parameters()),
                                              lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)

    def set_input(self, input):
        """从数据加载器解压缩输入数据并执行必要的预处理步骤.

        Parameters:
            input (dict): 包括数据本身及其元数据信息。
        """
        self.origin_img = input['origin_img'].to(self.device)
        self.wb_img = input['wb_img'].to(self.device)
        self.ce_img = input['ce_img'].to(self.device)
        self.gc_img = input['gc_img'].to(self.device)
        self.gt_img = input['gt_img'].to(self.device)
        self.image_paths = input['origin_img_paths']

    def forward(self):
        """前向传播;由两个函数调用 <optimize_parameters> and <test>."""
        self.generate_img = self.netWaterNet(self.origin_img, self.wb_img, self.ce_img, self.gc_img)

    def backward(self):
        """计算模型的损失"""

        lambda_A = self.opt.lambda_A  # 损失的权重           默认值为80
        lambda_B = self.opt.lambda_B  # 损失的权重           默认值为20
        # Identity loss
        self.loss_M = lambda_A * self.perceptual_loss(self.generate_img, self.gt_img) + lambda_B * self.network_loss(
            self.generate_img, self.gt_img)
        self.loss_M.backward()

    def optimize_parameters(self):
        """计算损失、梯度并更新网络权重;在每次训练迭代中调用"""
        # forward
        self.forward()  # 生成脱水图像.
        self.optimizer.zero_grad()  #
        self.backward()  # 计算梯度
        self.optimizer.step()  # 更新权重
