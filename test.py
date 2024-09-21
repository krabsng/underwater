"""用于图像到图像转换的通用测试脚本。

使用 train.py 训练模型后，可以使用此脚本来测试模型。
它将从'--checkpoints_dir' 加载模型 并 将结果保存到 '--results_dir'.

它首先在给定选项的情况下创建模型和数据集。它将对一些参数进行硬编码。
然后，它对“--num_test”图像运行推理，并将结果保存到 HTML 文件中。

示例（您需要先训练模型或从我们的网站下载预训练模型）:
    测试 CycleGAN 模型（两端）:
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    测试 CycleGAN 模型（仅限一侧）：
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    选项“--model test”仅用于生成一侧的 CycleGAN 结果。
    此选项将自动设置“--dataset_mode single”，仅从一组图像加载图像。
    相反，使用“--model cycle_gan”需要在两个方向上加载和生成结果,
    这有时是不必要的。结果将保存在 ./results/ 中。
    使用 '--results_dir <directory_path_to_save_result>' 指定结果所存放的目录。

    测试 pix2pix 模型:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

有关更多测试选项，请参阅 options/base_options.py 和 options/test_options.py。
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from utils.visualizer import save_images
from utils import html

try:
    import wandb
except ImportError:
    print('警告：找不到 wandb 包。选项“--use_wandb”将导致错误。')


if __name__ == '__main__':
    opt = TestOptions().parse()  # 获取测试选项
    # 对一些参数进行硬编码以进行测试
    opt.num_threads = 0   # 测试代码仅支持 num_threads = 0
    opt.batch_size = 1    # 测试代码仅支持 batch_size = 1
    opt.serial_batches = True  # 禁用数据洗牌;如果需要随机选择的图像结果，请注释此行。
    opt.no_flip = True    # 没有翻转;如果需要翻转图像的结果，请注释此行。
    opt.display_id = -1   # 无 visdom 显示;测试代码将结果保存到 HTML 文件中。
    dataset = create_dataset(opt)  # 创建给定opt.dataset_mode和其他选项的数据集
    model = create_model(opt)      # 在给定 opt.model 和其他选项的情况下创建模型
    model.setup(opt)               # 常规设置：加载和打印网络;创建调度程序;如果是继续训练或者测试,就加载模型权重;

    # 初始化 logger
    if opt.use_wandb:
        wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    # 创建网站
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # 定义网站目录
    if opt.load_iter > 0:  # load_iter默认为 0
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('创建 Web 目录', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # 使用 EVAL 模式进行测试。这只会影响 batchnorm 和 dropout 等层。
    # 对于 [pix2pix]：我们在原始 pix2pix 中使用 batchnorm 和 dropout。您可以在使用和不使用 eval（） 模式的情况下进行试验。
    # 对于 [CycleGAN]：它不应影响 CycleGAN，因为 CycleGAN 使用没有 dropout 的 instancenorm。
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # 仅将我们的模型应用于opt.num_test图像。
            break
        model.set_input(data)  # 从数据加载程序中解压缩数据
        model.test()           # 运行推理
        visuals = model.get_current_visuals()  # 获取图像结果
        img_path = model.get_image_paths()     # 获取图像路径
        if i % 5 == 0:  # 将图像保存到 HTML 文件
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
    webpage.save()  # 保存 HTML
