from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """本类包含训练选项。

    它还包括在 BaseOptions 中定义的共享选项。
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom 和 HTML 可视化参数
        parser.add_argument('--display_freq', type=int, default=25, help='在屏幕上显示训练结果的频率')
        parser.add_argument('--display_ncols', type=int, default=4, help='如果为正数，则在单个 Visdom Web 面板中显示所有图像，每行显示一定数量的图像。')
        parser.add_argument('--display_id', type=int, default=1, help='Web 显示的窗口 ID') # default设置成0就可以禁用visdom
        parser.add_argument('--display_server', type=str, default="http://localhost", help='Web 显示的 Visdom 服务器')
        parser.add_argument('--display_env', type=str, default='main', help='Visdom 显示环境名称（默认为 “main”）')
        parser.add_argument('--display_port', type=int, default=8097, help='Web 显示的 visdom 端口')
        parser.add_argument('--update_html_freq', type=int, default=1000, help='将训练结果保存到 HTML 的频率')
        parser.add_argument('--print_freq', type=int, default=100, help='在控制台上显示训练结果的频率')
        parser.add_argument('--no_html', action='store_true', help='不要将中级训练结果保存到 [opt.checkpoints_dir]/[opt.name]/web/')
        # 网络保存和加载参数f
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='保存最新结果的频率')
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='在epochs结束时保存检查点的频率')
        parser.add_argument('--save_by_iter', action='store_true', help='是否通过迭代保存模型')
        parser.add_argument('--continue_train', action='store_true', help='继续训练：加载最新模型')
        parser.add_argument('--epoch_count', type=int, default=1, help='起始纪元计数，我们按以下方式保存模型 <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # 训练参数
        parser.add_argument('--distributed', type=str, default=False, help='是否使用分布式训练')
        parser.add_argument('--pretrain_dir', type=str, default=None)
        parser.add_argument('--weight_decay', type=float, default=0.0005)
        parser.add_argument('--n_epochs', type=int, default=30, help='具有初始学习率的 epoch 数')
        parser.add_argument('--n_epochs_decay', type=int, default=0, help='线性衰减学习率为零的 epoch 数')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--gan_mode', type=str, default='lsgan', help='GAN 目标的类型。 [vanilla| lsgan | wgangp]. vanilla GAN损失是原始GAN论文中使用的交叉熵目标。')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lr_policy', type=str, default='cosine', help='学习率策略. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')

        self.isTrain = True
        return parser
