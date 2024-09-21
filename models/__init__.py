"""该软件包包含与目标函数、优化和网络架构相关的模块。

若要添加名为“dummy”的自定义模型类，需要添加名为“dummy_model.py”的文件，并定义继承自 BaseModel 的子类 DummyModel。
您需要实现以下五个函数：
    -- <__init__>:                      初始化类;首先调用 BaseModel.__init__（self， opt）。
    -- <set_input>:                     从数据集中解压缩数据并应用预处理。
    -- <forward>:                       产生中间结果.
    -- <optimize_parameters>:           计算损失、梯度并更新网络权重。
    -- <modify_commandline_options>:    （可选）添加特定于模型的选项并设置默认选项。

在函数<__init__>中，您需要定义四个列表：
    -- self.loss_names (str list):          指定要绘制和保存的训练损失。
    -- self.model_names (str list):         定义我们训练中使用的网络。
    -- self.visual_names (str list):        指定要显示和保存的图像。
    -- self.optimizers (optimizer list):    定义和初始化优化器。 您可以为每个网络定义一个优化器。 如果两个网络同时更新，可以使用 itertools.chain 对它们进行分组。有关用法，请参阅cycle_gan_model.py。

现在，您可以通过指定标志“--model dummy”来使用模型类。
有关详细信息，请参阅我们的模板模型类“template_model.py”。
"""

import importlib
from models.base_model import BaseModel


def find_model_using_name(model_name):
    """导入模块“models/[model_name]_model.py”。

    在该文件中，名为 DatasetNameModel（） 的类将
    被实例化。它必须是 BaseModel 的子类，
    并且不区分大小写。
    """
    model_filename = "models." + model_name + "_model"
    modellib = importlib.import_module(model_filename) # importlib模块用于动态加载其他模块
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model


def get_option_setter(model_name):
    """返回<modify_commandline_options>模型类的静态方法。"""
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options


def create_model(opt):
    """Create a model given the option.

    This function warps the class CustomDatasetDataLoader.
    This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from models import create_model
        >>> model = create_model(opt)
    """
    model = find_model_using_name(opt.model)
    instance = model(opt)
    print("model [%s] was created" % type(instance).__name__)
    return instance
