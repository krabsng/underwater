import argparse
import os
from utils import util
import torch
import models
import data


class BaseOptions():
    """此类定义在训练和测试期间使用的选项。

    它还实现了几个帮助程序函数，例如解析、打印和保存选项。
    它还收集了<modify_commandline_options>在数据集类和模型类的函数中定义的其他选项。
    """

    def __init__(self):
        """重置类;指示类尚未初始化"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # 基本参数
        parser.add_argument('--dataroot', default="/a.krabs/dataset/EUVP/Paired/",
                            help='图像路径（应具有子文件夹 trainA、trainB、valA、valB 等）') # 在AI主机上训练设置的参数
        parser.add_argument('--name', type=str, default='spu_net',
                            help='实验的名称。它决定将样本和模型存储在何处')
        parser.add_argument('--gpu_ids', type=str, default='0,1,2,3', help='GPU ID：例如 0 0,1,2， 0,2。使用 -1 表示 CPU') # 在AI主机上训练设置的参数
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='模型保存在此处')
        # 模型参数
        parser.add_argument('--model', type=str, default='spu',
                            help='chooses which model to use. [cycle_gan | pix2pix | test | colorization | krabs]')
        parser.add_argument('--input_nc', type=int, default=3, help='# 输入图像通道数：RGB 为 3 个，灰度为 1 个')
        parser.add_argument('--output_nc', type=int, default=3, help='# 输出图像通道数：RGB 为 3 个，灰度为 1 个')
        parser.add_argument('--ngf', type=int, default=64, help='# 最后一个卷积层中的 gen 滤波器')
        parser.add_argument('--ndf', type=int, default=64, help='# 第一个卷积层中的离散滤波器')
        parser.add_argument('--netD', type=str, default='basic',
                            help='指定判别器的结构 [basic | n_layers | pixel]. basic是 70x70 PatchGAN。n_layers允许您在鉴别器中指定层')
        parser.add_argument('--netG', type=str, default='resnet_9blocks',
                            help='指定生成器的结构 [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
        parser.add_argument('--n_layers_D', type=int, default=3, help='仅当 netD==n_layers 生效')
        parser.add_argument('--norm', type=str, default='instance',
                            help='实例规范化或批量规范化 [instance | batch | none]')
        parser.add_argument('--init_type', type=str, default='normal',
                            help='网络初始化 [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02,
                            help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--no_dropout', action='store_true', help='生成器未使用dropout')
        # 数据集参数
        parser.add_argument('--dataset_mode', type=str, default='sr',
                            help='选择数据集的加载方式。 [unaligned | aligned | single | colorization | underwater | sr]')
        parser.add_argument('--direction', type=str, default='AtoB', help='AtoB 或 BtoA')
        parser.add_argument('--serial_batches', action='store_true',
                            help='如果为 true，则拍摄图像以进行批量处理，否则随机拍摄图像')
        parser.add_argument('--num_threads', default=12, type=int, help='# 用于加载数据的线程数量') # 更改为0
        parser.add_argument('--batch_size', type=int, default=16, help='输入的批量大小') # 在AI主机上训练设置的参数
        parser.add_argument('--load_size', type=int, default=256, help='将图像缩放到此大小')
        parser.add_argument('--crop_size', type=int, default=256, help='然后裁剪到这个大小')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                            help='每个数据集允许的最大样本数。如果数据集目录包含多个max_dataset_size，则仅加载一个子集。')
        parser.add_argument('--preprocess', type=str, default='resize_and_crop',
                            help='加载时缩放和裁剪图像 [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
        parser.add_argument('--no_flip', action='store_true', help='如果指定，请不要翻转图像以进行数据增强')
        parser.add_argument('--display_winsize', type=int, default=256, help='visdom 和 HTML 的显示窗口大小')
        # 其他参数
        parser.add_argument('--epoch', type=str, default='latest',
                            help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--load_iter', type=int, default='0',
                            help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        parser.add_argument('--verbose', action='store_true', help='如果指定，请打印更多调试信息')
        parser.add_argument('--suffix', default='', type=str,
                            help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
        # WANDB 参数
        parser.add_argument('--use_wandb', action='store_true', help='如果指定，则初始化 wandb 日志记录')
        parser.add_argument('--wandb_project_name', type=str, default='CycleGAN-and-pix2pix',
                            help='specify wandb project name')
        self.initialized = True
        return parser

    def gather_options(self):
        """使用基本选项初始化我们的解析器（仅一次）。
        添加其他特定于模型和特定于数据集的选项。
        这些选项在函数中定义<modify_commandline_options>
        在模型和数据集类中。
        """
        if not self.initialized:  # 检查是否已初始化
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # 获取基本选项
        opt, _ = parser.parse_known_args()

        # 修改与模型相关的解析器选项
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # 使用新的默认值再次解析

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        # 保存并返回解析器
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        if opt.isTrain:
            message += '{:>25}:{:<30}\n'.format('visdom服务器地址', opt.display_server + ':' + str(opt.display_port))
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        """解析我们的选项，创建检查点目录后缀，并设置 gpu 设备。"""
        opt = self.gather_options()
        opt.isTrain = self.isTrain  # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
