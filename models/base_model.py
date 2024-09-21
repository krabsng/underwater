import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import networks


class BaseModel(ABC):
    """此类是模型的抽象基类 （ABC）。
    要创建子类，您需要实现以下五个函数：
        -- <__init__>:                      初始化类;首先调用 BaseModel.__init__（self， opt）。
        -- <set_input>:                     从数据集中解压缩数据并应用预处理。
        -- <forward>:                       产生中间结果。
        -- <optimize_parameters>:           计算损失、梯度并更新网络权重。
        -- <modify_commandline_options>:    （可选）添加特定于模型的选项并设置默认选项。
    """

    def __init__(self, opt):
        """初始化 BaseModel 类。

        Parameters:
            opt (Option class) -- 存储所有实验标志;需要是 BaseOptions 的子类

        在创建自定义类时，您需要实现自己的初始化。
        在这个函数中，你应该首先调用 <BaseModel.__init__（self， opt）>
        然后，您需要定义四个列表：
            -- self.loss_names (str list):          指定要绘制和保存的训练损失。
            -- self.model_names (str list):         定义我们训练中使用的网络。
            -- self.visual_names (str list):        指定要显示和保存的图像。
            -- self.optimizers (optimizer list):    定义和初始化优化器。 您可以为每个网络定义一个优化器。 如果两个网络同时更新, 您可以使用 itertools。链以对它们进行分组。有关示例，请参阅 cycle_gan_model.py。
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # 获取设备名称：CPU 或 GPU
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # 将所有检查点保存到save_dir
        if opt.preprocess != 'scale_width':  # 使用 [scale_width]，输入图像可能具有不同的大小，这会损害 cudnn.benchmark 的性能。
            torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # 用于学习率策略 'plateau'

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """添加新的特定于模型的选项，并重写现有选项的默认值。

        Parameters:
            parser          -- 原始选项解析器
            is_train (bool) -- 无论是训练阶段还是测试阶段。可以使用此标志添加特定于训练或特定于测试的选项。

        Returns:
            修改后的解析器。
        """
        return parser

    @abstractmethod
    def set_input(self, input):
        """从数据加载器中解压缩输入数据，并执行必要的预处理步骤。

        Parameters:
            输入 （dict）：包括数据本身及其元数据信息。
        """
        pass

    @abstractmethod
    def forward(self):
        """前向传播;由<optimize_parameters>和<test>调用。"""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """计算损失、梯度并更新网络权重;在每次训练迭代中调用"""
        pass

    def setup(self, opt):
        """加载和打印网络;创建调度程序

        参数：
            opt （Option 类） -- 存储所有实验标志;需要是 BaseOptions 的子类
        """
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.isTrain or opt.continue_train:
            load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            self.load_networks(load_suffix)
        self.print_networks(opt.verbose)

    def eval(self):
        """在测试期间使模型成为评估模式"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def test(self):
        """测试时使用的前向传播函数。

        此函数将<forward>函数包装在 no_grad（） 中，因此我们不会为 backprop 保存中间步骤
        它还调用<compute_visuals>以生成其他可视化结果
        """
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        """计算 visdom 和 HTML 可视化的其他输出图像"""
        pass

    def get_image_paths(self):
        """ 返回用于加载当前数据的图像路径"""
        return self.image_paths

    def update_learning_rate(self):
        """更新所有网络的学习率;在每个纪元结束时被调用"""
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))

    def get_current_visuals(self):
        """返回可视化图像。train.py 将使用 visdom 显示这些图像，并将图像保存为 HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        """返回训练损失/错误。train.py 将在控制台上打印出这些错误或损失，并将它们保存到文件中"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def save_networks(self, epoch):
        """将所有网络保存到磁盘。

        Parameters:
            epoch (int) -- 当前epoch;使用 '%s_net_%s.pth' % （epoch， name） 作为文件名
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    if isinstance(net, torch.nn.DataParallel):
                        # 如果是 DataParallel，使用 .module 访问实际模型
                        torch.save(net.module.cpu().state_dict(), save_path)
                    else:
                        # 如果不是 DataParallel，直接保存模型
                        torch.save(net.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """修复 InstanceNorm 检查点不兼容（0.4 之前）"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, epoch):
        """从磁盘加载所有网络。

        参数：
            epoch （int） -- 当前epoch;用于文件名 '%s_net_%s.pth' % （epoch， 名称）
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('从 %s 加载网络模型' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    def print_networks(self, verbose):
        """打印网络中的参数总数和（如果详细）网络架构

        Parameters:
            verbose (bool) -- 如果详细：打印网络体系结构
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """为所有网络设置 requies_grad=Fasle，以避免不必要的计算
        Parameters:
            nets (network list)   -- 网络列表
            requires_grad (bool)  -- 网络是否需要梯度
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
