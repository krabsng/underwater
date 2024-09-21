# krabs自定义的数据集，用于超分辨率从水下图像增强
"""数据集类模板

该模块为用户提供了实现自定义数据集的模板。
您可以指定“--dataset_mode模板”来使用此数据集。
类名应与文件名及其dataset_mode选项一致。
文件名应为 <dataset_mode>_dataset.py
类名应<Dataset_mode>为 Dataset.py
您需要实现以下函数：
    -- <modify_commandline_options>:　添加特定于数据集的选项，并重写现有选项的默认值。
    -- <__init__>: 初始化此数据集类。
    -- <__getitem__>: 返回数据点及其元数据信息.
    -- <__len__>: 返回图像数量.
"""
from PIL import Image
from data.base_dataset import BaseDataset, get_params, get_transform
from utils import util
import torchvision.transforms as transforms


class SRDataset(BaseDataset):
    """用于实现自定义数据集的模板数据集类。"""

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """添加特定于数据集的新选项，并重写现有选项的默认值.

        参数：
            parser          -- original option parser
            is_train (bool) -- 无论是训练阶段还是测试阶段。可以使用此标志添加特定于训练或特定于测试的选项.

        Returns:
            the modified parser.
        """
        # parser.add_argument('--new_dataset_option', type=float, default=1.0, help='new dataset option')
        # 这里由于需要使用超分辨率的数据，所以修改了dataroot默认值
        parser.set_defaults(dataroot="/a.krabs/dataset/UFO120/train_val/")
        # 数组增强的方式
        parser.set_defaults(preprocess='none')
        # visdom一行显示几张图片，这里设置成0，否则会出现问题
        parser.set_defaults(display_ncols=0)
        return parser

    def __init__(self, opt):
        """初始化此数据集类。

        Parameters:
            opt (Option class) -- 存储所有实验标志;需要是 BaseOptions 的子类

        这里可以做一些事情。
        - 保存选项（已在 BaseDataset 中完成）
        - 获取数据集的图像路径和元信息。
        - 定义图像转换。
        """
        self.isTrain = opt.isTrain
        # 保存选项和数据集根目录
        BaseDataset.__init__(self, opt)
        # 模型训练和验证加载的数据不一样，所以在这里进行额外处理
        if self.isTrain is None:
            self.data_list = util.populate_train_list(opt.dataroot, mode="test", pattern="*.jpg")
        elif self.isTrain:
            self.data_list = util.populate_train_list(opt.dataroot, mode='train', pattern='lrd/*.jpg')
        else:  # test_options修改了dataroot的值
            self.data_list = util.populate_train_list(opt.dataroot, mode='val', pattern='Inp/*.jpg')
        self.input_nc = self.opt.input_nc
        self.output_nc = self.opt.output_nc

        # self.Apaths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # 获取图像A的所有路径
        # self.Bpaths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  # 获取图像B的所有路径

    def __getitem__(self, index):
        """返回数据点及其元数据信息。

        Parameters:
            index -- 用于数据索引的随机整数

        Returns:
           带有其名称的数据字典。它通常包含数据本身及其元数据信息。

        第 1 步：获取随机图像路径：例如, path = self.image_paths[index]
        第 2 步：从磁盘加载数据：例如, image = Image.open(path).convert('RGB').
        第 3 步：将数据转换为 PyTorch 张量。 您可以使用帮助程序函数，例如 self.transform。例如, data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        # 读取给定随机整数索引的图像
        origin_img_path = self.data_list[index]
        gt_img_path = origin_img_path.replace('lrd', 'hr')
        origin_img = Image.open(origin_img_path).convert('RGB')
        gt_img = Image.open(gt_img_path).convert('RGB')
        # # 在这里对图像进行随机翻转,裁减
        # or_transform_params = get_params(self.opt, origin_img.size)  # 获取随机裁减和翻转的参数
        # gt_transform_params = get_params(self.opt, gt_img.size)  # 获取随机裁减和翻转的参数
        # origin_img_transform = get_transform(self.opt, or_transform_params, grayscale=(self.input_nc == 1),convert=True)
        # gt_img_transform = get_transform(self.opt, gt_transform_params, grayscale=(self.output_nc == 1), convert=True)
        # origin_img = origin_img_transform(origin_img)
        # gt_img = gt_img_transform(gt_img)
        transformer = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        origin_img = transformer(origin_img)
        gt_img = transformer(gt_img)
        return {'A': origin_img, 'B': gt_img, 'A_paths': origin_img_path, 'B_paths': gt_img_path}

    def __len__(self):
        """返回图像总数。"""
        return len(self.data_list)
