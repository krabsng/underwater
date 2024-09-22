"""用于图像到图像转换的通用训练脚本。

此脚本适用于各种模型(使用选项“--model”：例如, pix2pix, cyclegan, colorization) 和
不同的数据集 (使用选项“--dataset_mode”：例如, aligned, unaligned, single, colorization).
您需要指定数据集('--dataroot'), 实验名称 ('--name'), 和模型 ('--model').

它首先在给定选项的情况下创建模型、数据集和可视化工具。
然后，它执行标准网络训练。在训练过程中，它还可视化/保存图像、打印/保存损失图和保存模型。
该脚本支持继续/恢复训练。使用“--continue_train”恢复之前的训练。

例：
    训练 CycleGAN 模型：
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    训练 pix2pix 模型：
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

有关更多培训选项，请参阅 options/base_options.py 和 options/train_options.py。
请参阅以下位置的培训和测试提示：https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
请参阅以下位置的常见问题： https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from utils.visualizer import Visualizer
from utils import utils

if __name__ == '__main__':
    opt = TrainOptions().parse()   # 获取训练选项
    if opt.opt.distributed:
        utils.init_distributed_mode(opt) # 使用分布式训练
    dataset = create_dataset(opt)  # 创建给定opt.dataset_mode和其他选项的数据集
    dataset_size = len(dataset)    # 获取数据集中的图像数。
    print('训练集图像的数量 = %d' % dataset_size)

    model = create_model(opt)      # 在给定 opt.model 和其他选项的情况下创建模型
    model.setup(opt)               # 常规设置：加载和打印网络;创建调度程序
    visualizer = Visualizer(opt)   # 创建显示/保存图像和绘图的可视化工具
    total_iters = 0                # 目前训练已迭代的总数

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # 不同代的外循环;我们通过 <epoch_count>， <epoch_count>+ <save_latest_freq>保存模型
        epoch_start_time = time.time()  # 整个代的计时器
        iter_data_time = time.time()    # 每次迭代数据加载的计时器
        epoch_iter = 0                  # 当前 epoch 中的训练迭代次数，每个 epoch 重置为 0
        visualizer.reset()              # 重置可视化工具：确保它至少在每个epoch将结果保存到 HTML 一次
        model.update_learning_rate()    # 在每个epoch 开始时更新学习率。
        for i, data in enumerate(dataset):  # 一个epoch内的内循环
            iter_start_time = time.time()  # 每次迭代计算的计时器
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # 从数据集中解压缩数据并应用预处理,数据的shape：[8,3,256,256]
            model.optimize_parameters()   # 计算损失函数，获取梯度，更新网络权重

            if total_iters % opt.display_freq == 0:   # 在 visdom 上显示图像并将图像保存到 HTML 文件
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # 打印训练损失并将日志记录信息保存到磁盘
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # 每次迭代都缓存最新模型<save_latest_freq>
                print('保存最新的模型： (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # 每隔一个时期缓存我们的模型<save_epoch_freq>
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
