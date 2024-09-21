"""该类用于使用指定模型，生成增强后的图像
"""
import os.path
import urllib.request
from pathlib import Path
from utils.util import classify_input as CLS
from utils.util import mkdir
from utils import util
from options.generate_options import GenerateOptions
from data import create_dataset
from models import create_model
from utils.visualizer import save_images
from utils import html

class WaterEnhancement():
    def __init__(self, model_name: str = "", input: str = ""):
        """创建模型
                parameters： model_name --模型的名称,
        """
        self.opt = GenerateOptions().parse()
        self.process_opt(self.opt, model_name, input)
        self.dataset = create_dataset(self.opt)  # 创建给定opt.dataset_mode和其他选项的数据集
        self.model = create_model(self.opt)  # 在给定 opt.model 和其他选项的情况下创建模型
        self.model.setup(self.opt)  # 常规设置：加载和打印网络;创建调度程序;如果是继续训练或者测试,就加载模型权重;
        self.model.eval() # 不需要计算梯度


    def predict(self):
        # 创建网站
        web_dir = os.path.join(self.opt.results_dir, self.opt.name, '{}_{}'.format(self.opt.phase, self.opt.epoch))  # 定义网站目录
        if self.opt.load_iter > 0:  # load_iter默认为 0
            web_dir = '{:s}_iter{:d}'.format(web_dir, self.opt.load_iter)
        print('创建 Web 目录', web_dir)
        webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (self.opt.name, self.opt.phase, self.opt.epoch))
        for i, data in enumerate(self.dataset):
            self.model.set_input(data)  # 从数据加载程序中解压缩数据
            self.model.test()  # 运行推理
            visuals = self.model.get_current_visuals()  # 获取图像结果
            img_path = self.model.get_image_paths()  # 获取图像路径
            if i % 5 == 0:  # 将图像保存到 HTML 文件
                print('processing (%04d)-th image... %s' % (i, img_path))
            save_images(webpage, visuals, img_path, aspect_ratio=self.opt.aspect_ratio, width=self.opt.display_winsize,
                        use_wandb=self.opt.use_wandb)
        webpage.save()  # 保存 HTML
    @staticmethod
    def process_opt(opt, model, input):
        dir_name = "./tmp"
        mkdir(dir_name)
        cls = CLS(input)
        if cls == "URL":
            downloaded_file = os.path.join(dir_name, Path(input).name)
            urllib.request.urlretrieve(input, downloaded_file)
            if CLS(downloaded_file) == "Image":
                dir_name = str(Path(downloaded_file).parent.resolve()) + "/"
            else:
                dir_name = str(Path(downloaded_file).parent.resolve()) + "/"
                util.video_to_frames(downloaded_file, dir_name)
        elif cls == "Folder":
            dir_name = input
        elif cls == "Image":
            dir_name = str(Path(input).parent.resolve()) + "/"
        elif cls == "Video":
            dir_name = str(Path(input).parent.resolve())
            util.video_to_frames(input, dir_name)
        else:
            raise ValueError("参数错误")
        opt.dataroot = dir_name
        opt.model = model
        opt.num_threads = 0  # 测试代码仅支持 num_threads = 0
        opt.batch_size = 1  # 测试代码仅支持 batch_size = 1
        opt.serial_batches = True  # 禁用数据洗牌;如果需要随机选择的图像结果，请注释此行。
        opt.no_flip = True  # 没有翻转;如果需要翻转图像的结果，请注释此行。
        opt.display_id = -1  # 无 visdom 显示;测试代码将结果保存到 HTML 文件中。

if __name__ == "__main__" :
    w = WaterEnhancement("iat", "/home/ljp/a.krabs/krabs/tmp/frame_0000.jpg")
    w.predict()
    # WaterEnhancement.process_opt(TestOptions().parse(), "aa", "https://k.sinaimg.cn/n/sinakd20108/699/w690h809/20240204/d5e1-254d7f50e3f31c04abc59ce72a71faae.jpg/w700d1q75cms.jpg")
