import torch
import itertools
from utils.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from loss import ssim_loss


class CycleGANModel(BaseModel):
    """
   此类实现了 CycleGAN 模型，用于在没有配对数据的情况下学习图像到图像的转换。

    模型训练需要“--dataset_mode unaligned”数据集。
    默认情况下，它使用“--netG resnet_9blocks”ResNet 生成器，
    一个 '--netD basic' 鉴别器（pix2pix 引入的 PatchGAN），
    以及最小二乘 GAN 目标 （'--gan_mode lsgan'）。

    CycleGAN原论文地址：https://arxiv.org/pdf/1703.10593.pdf 2017年发布
    """

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
            parser.add_argument('--lambda_identity', type=float, default=0.5,  # 这里将idt损失置零
                                help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. 例如，如果身份损失的权重应小于重建损失的权重的 10 倍，请将 lambda_identity = 0.1')
            parser.add_argument('--lambda_ssim', type=float, default=1)

        return parser

    def __init__(self, opt):
        """初始化 CycleGAN 类。

        Parameters:
            opt (Option class)-- 存储所有实验标志;需要是 BaseOptions 的子类
        """
        BaseModel.__init__(self, opt)
        # 指定要打印的训练损失. 训练/测试脚本将调用 <BaseModel.get_current_losses>
        """损失函数的构成
                判别器D_A的损失 loss_D_A = ||D_A(G_A(real_A)) - False|| + ||D_A(real_B) - True||
                生成器G_A的损失 loss_G_A = ||D_A(G_A(real_A)) - True||
                循环一致性损失  loss_cycle_A = ||G_B(G_A(real_A)) - real_A||
                恒等损失       loss_idt_A = ||G_A(real_B) - real_B||
            
            四类不同的图像
                real_A:真实的图片   fake_B: 生成器G_A生成的图片   rec_A : G_B(G_A(A))   idt_B : G_A(B)
                real_B:真实的图片   fake_A: 生成器G_B生成的图片   rec_B : G_A(G_B(B))   idt_A : G_B(A)
                
        """
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        # 指定要保存/显示的图像. 训练/测试脚本将调用 <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # 如果使用了idt损失, 我们还将 idt_B=G_A（B） 和 idt_A=G_A（B） 可视化
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # 合并 A 和 B 的可视化效果
        # 指定要保存到磁盘的模型. 训练/测试脚本将调用<BaseModel.save_networks> 和 <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # 在测试期间，仅加载 Gs
            self.model_names = ['G_A', 'G_B']

        # 定义网络（生成器和判别器）
        # 命名与论文中使用的命名不同。
        # 代码 (vs. 论文): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # 定义判别器
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # 仅当输入和输出图像具有相同数量的通道时才有效
                assert (opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # 创建图像缓冲区以存储以前生成的图像
            self.fake_B_pool = ImagePool(opt.pool_size)  # 创建图像缓冲区以存储以前生成的图像
            # 定义损失函数
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # 定义GAN损失.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()

            # 初始化优化器;调度程序将由函数 <BaseModel.setup> 自动创建。
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
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
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """前向传播;由两个函数调用 <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)  # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)  # G_A(G_B(B))

    def backward_D_basic(self, netD, real, fake):
        """计算鉴别器的GAN损失

        参数：
            netD (network)      -- 判别器 D
            real (tensor array) -- 真实图像
            fake (tensor array) -- 生成器生成的图像

        返回判别器损失。
        我们还调用 loss_D.backward() 来计算梯度。  优化器的step()方法是用来更新梯度的
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)  # 此时 gan_mode = lsgan 损失函数为 nn.MSELoss()
        # Fake
        pred_fake = netD(fake.detach())  # detach()函数是PyTorch张量的的方法之一,其作用是创建一个与原始张量相同的数据,但不具备梯度
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # 组合损失并计算梯度
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """计算判别器D_A的GAN损失
            生成器G_A(real_a)生成fake_b
            判别器D_A判别real_b和fake_b
        """
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """计算判别器D_B的GAN损失"""
        fake_A = self.fake_A_pool.query(self.fake_A)  # fake_A_pool是图像缓冲区
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """计算生成器G_A和G_B的损失"""

        lambda_idt = self.opt.lambda_identity  # 损失的权重 默认值为0.5 现在为0
        lambda_A = self.opt.lambda_A  # 损失的权重           默认值为10  现在为1
        lambda_B = self.opt.lambda_B  # 损失的权重           默认值为10  现在为1
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B),
                                          True)  # 生成器G_A的损失 loss_G_A = ||D_A(G_A(real_A)) - True|| 就是判别器判别生成器G_A生成的fake_B是real_B的概率
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A,
                                                self.real_A) * lambda_A  # 循环一致性损失 loss_cycle_A = ||G_B(G_A(real_A)) - real_A||
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

    def optimize_parameters(self):
        """计算损失、梯度并更新网络权重;在每次训练迭代中调用"""
        # forward
        self.forward()  # 计算假图像和重建图像.
        # G_A 和 G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds 在优化时， Gs 不需要梯度
        self.optimizer_G.zero_grad()  # 将 G_A 和 G_B 的梯度设置为零
        self.backward_G()  # 计算 G_A 和 G_B 的梯度
        self.optimizer_G.step()  # 更新G_A和G_B权重
        # D_A 和 D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()  # 将 D_A 和 D_B 的梯度设置为零
        self.backward_D_A()  # 计算D_A梯度
        self.backward_D_B()  # 计算 D_B 的梯度
        self.optimizer_D.step()  # 更新D_A和D_B权重
