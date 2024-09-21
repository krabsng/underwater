from .base_options import BaseOptions


class GenerateOptions(BaseOptions):
    """此类包括测试选项。

    它还包括在 BaseOptions 中定义的共享选项。
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # 定义共享选项
        parser.add_argument('--results_dir', type=str, default='./generates/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout 和 Batchnorm 在训练和测试期间具有不同的行为。
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=500, help='how many test images to run')
        # rewrite devalue values
        parser.set_defaults(name='iat')
        parser.set_defaults(model='iat')
        # 为避免裁剪，load_size应与crop_size相同
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        parser.set_defaults(preprocess='none')
        # 这里由于测试集和训练集的数据不同，所以修改了dataroot默认值
        parser.set_defaults(dataroot="")
        self.isTrain = None
        return parser
