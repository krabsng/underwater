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
from data.base_dataset import BaseDataset, get_transform
# from data.image_folder import make_dataset
# from PIL import Image


class TemplateDataset(BaseDataset):
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
        parser.set_defaults(max_dataset_size=10, new_dataset_option=2.0)  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        """初始化此数据集类。

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        这里可以做一些事情。
        - 保存选项（已在 BaseDataset 中完成）
        - 获取数据集的图像路径和元信息。
        - 定义图像转换。
        """
        # 保存选项和数据集根目录
        BaseDataset.__init__(self, opt)
        #获取数据集的图像路径;
        self.image_paths = []  # 你可以调用sorted（make_dataset（self.root， opt.max_dataset_size）））来获取self.root目录下的所有镜像路径
        # 定义默认转换函数。你可以使用 <base_dataset.get_transform>; 您还可以定义自定义转换函数
        self.transform = get_transform(opt)

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
        path = 'temp'    # 需要是字符串
        data_A = None    # 需要是张量
        data_B = None    # 需要是张量
        return {'data_A': data_A, 'data_B': data_B, 'path': path}

    def __len__(self):
        """返回图像总数。"""
        return len(self.image_paths)
