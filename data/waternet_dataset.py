# krabs自定义的数据集，为waternet加载水下数据
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
import numpy as np
from PIL import Image
from data.base_dataset import BaseDataset, get_params, get_transform
from utils import util
from physics_models import WB, CLAHE, gamma

class WaterNetDataset(BaseDataset):
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
        parser.add_argument('--new_dataset_option', type=float, default=1.0, help='new dataset option')
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
            self.data_list = util.populate_train_list(opt.dataroot, mode='train', pattern='*/trainA/*.jpg')
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
        # 训练和验证模型所使用的数据不一样，在这里做额外处理
        if self.isTrain is None:
            origin_img = Image.open(origin_img_path).convert('RGB')
            transform_params = get_params(self.opt, origin_img.size)  # 获取随机裁减和翻转的参数
            origin_img_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1),
                                                 convert=True)
            origin_img = origin_img_transform(origin_img)
            return {'A': origin_img, 'A_paths': origin_img_path}
        elif self.isTrain:
            gt_img_path = origin_img_path.replace('trainA', 'trainB')
        else:
            gt_img_path = origin_img_path.replace('Inp', 'GTr')
        """
            origin_img:表示原始的 RGB 图像
            wb_img:原始图像经白平衡处理后的图像
            ce_img：直方图处理后的图像
            gc_img：伽马矫正后的图像
            gt_img：表示目标图像
        """
        origin_img = Image.open(origin_img_path).convert('RGB')
        wb_img = Image.fromarray(WB.white_balance(np.array(origin_img)))
        ce_img = Image.fromarray(CLAHE.apply_clahe_to_color_image(np.array(origin_img)))
        gc_img = Image.fromarray(gamma.gamma_correction(np.array(origin_img), 2.2))
        gt_img = Image.open(gt_img_path).convert('RGB')
        # 在这里对图像进行随机翻转,裁减
        transform_params = get_params(self.opt, origin_img.size)  # 获取随机裁减和翻转的参数
        origin_img_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1),convert=True)
        gt_img_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1), convert=True)
        origin_img = origin_img_transform(origin_img)
        wb_img = origin_img_transform(wb_img)
        ce_img = origin_img_transform(ce_img)
        gc_img = origin_img_transform(gc_img)
        gt_img = gt_img_transform(gt_img)

        """以下代码是源代码对图像的处理，存在些许问题
        # origin_img, gt_img = (np.asarray(origin_img) / 255.0), (np.asarray(gt_img) / 255.0)
        # # 进行标准化
        # transform_input = Compose(
        #     [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ConvertImageDtype(torch.float), ])
        # transform_gt = Compose([ToTensor(), ConvertImageDtype(torch.float), ])
        A, B = transform_input(origin_img), transform_gt(gt_img)
        """
        return {'origin_img': origin_img,
                'wb_img': wb_img,
                'ce_img': ce_img,
                'gc_img': gc_img,
                'gt_img': gt_img,
                'origin_img_paths': origin_img_path,
                'gt_img_paths': gt_img_path
                }

    def __len__(self):
        """返回图像总数。"""
        return len(self.data_list)
