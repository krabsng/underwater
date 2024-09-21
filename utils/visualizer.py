import numpy as np
import os
import sys
import ntpath
import time
from . import util, html
from subprocess import Popen, PIPE

try:
    import wandb
except ImportError:
    print('警告：找不到 wandb 包。选项“--use_wandb”将导致错误。')
# 检查python解释器的版本号
if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256, use_wandb=False):
    """将图像保存到磁盘.

    Parameters:
        webpage (the HTML class) -- 存储这些 imaeg 的 HTML 网页类（有关详细信息，请参见 html.py）
        visuals (OrderedDict)    -- 存储（名称、图像（张量或 NumPy））对的有序字典
        image_path (str)         -- 该字符串用于创建图像路径
        aspect_ratio (float)     -- 保存图像的纵横比
        width (int)              -- 图像的大小将调整为宽度 x 宽度

   此功能会将存储在“visuals”中的图像保存到“webpage”指定的HTML文件中。
    """
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []
    ims_dict = {}
    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        util.save_image(im, save_path, aspect_ratio=aspect_ratio)
        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
        if use_wandb:
            ims_dict[label] = wandb.Image(im)
    webpage.add_images(ims, txts, links, width=width)
    if use_wandb:
        wandb.log(ims_dict)


class Visualizer():
    """此类包含几个函数，可以显示/保存图像和打印/保存日志记录信息。

    它使用 Python 库“visdom”进行显示，并使用 Python 库“dominate”（包装在“HTML”中）创建带有图像的 HTML 文件。
    """

    def __init__(self, opt):
        """初始化 Visualizer 类

        Parameters:
            opt -- 存储所有实验标志;需要是 BaseOptions 的子类
        Step 1: 缓存训练/测试选项
        Step 2: 连接到 Visdom 服务器
        Step 3: 创建一个 HTML 对象以保存 HTML 过滤器
        Step 4: 创建日志记录文件以存储训练损失
        """
        self.opt = opt  # 缓存选项
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.port = opt.display_port
        self.saved = False
        self.use_wandb = opt.use_wandb
        self.wandb_project_name = opt.wandb_project_name
        self.current_epoch = 0
        self.ncols = opt.display_ncols
        if self.display_id > 0:  # 连接到给定的 Visdom 服务器<display_port>，然后<display_server>
            import visdom
            # 创建visdom.server,重复创建不会引发问题
            self.create_visdom_connections()
            self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, env=opt.display_env)
            # 检查是否连接到visdom服务器,如果没连接上,则重新连接
            # if not self.vis.check_connection():
            #     self.create_visdom_connections()
        if self.use_wandb:
            self.wandb_run = wandb.init(project=self.wandb_project_name, name=opt.name,
                                        config=opt) if not wandb.run else wandb.run
            self.wandb_run._label(repo='CycleGAN-and-pix2pix')

        if self.use_html:  # 在 <checkpoints_dir>/web/ 创建一个 HTML 对象; 图片将保存在 <checkpoints_dir>/web/images/ 下
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        # 创建日志记录文件以存储训练损失
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """重置 self.saved 的状态"""
        self.saved = False

    def create_visdom_connections(self):
        """此功能将在端口 self.port 启动一个visdom服务器,重复启动不会引发问题>"""
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        print('\n\n尝试启动visdom的服务用于展示训练数据....')
        print('命令: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def display_current_results(self, visuals, epoch, save_result):
        """在 visdom 上显示当前结果;将当前结果保存到 HTML 文件。

        Parameters:
            visuals (OrderedDict) -- 要显示或保存的图像字典,包含图像的名称和图像数据本身
            epoch (int) -- 当前epoch
            save_result (bool) -- 是否将当前结果保存到 HTML 文件中
        """
        if self.display_id > 0:  # 使用 Visdom 在浏览器中显示图像
            #
            ncols = self.ncols
            if ncols > 0:  # 在一个 Visdom 面板中显示所有图像
                # 每行显示多少个数据
                ncols = min(ncols, len(visuals))
                # iter()函数获取数据的迭代器 nex()函数取数据 可能需要根据网络模型进行修改
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing: 4px; white-space: nowrap; text-align: center}
                        table td {width: % dpx; height: % dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)  # create a table css
                # 创建图像表。
                title = self.name
                label_html = ''   # 对应的标签名称,以一个表的形式展示
                label_html_row = ''
                images = []
                idx = 0
                for label, image in visuals.items():
                    image_numpy = util.tensor2im(image)
                    label_html_row += '<td>%s</td>' % label
                    # 可能需要根据网络模型进行修改
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                try:
                    self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                    padding=2, opts=dict(title=title + ' images'))
                    label_html = '<table>%s</table>' % label_html
                    self.vis.text(table_css + label_html, win=self.display_id + 2,
                                  opts=dict(title=title + ' labels'))
                except VisdomExceptionBase:
                    self.create_visdom_connections()

            else:  # 在单独的 Visdom 面板中显示每张图片;
                idx = 1
                try:
                    for label, image in visuals.items():
                        image_numpy = util.tensor2im(image)
                        self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                       win=self.display_id + idx)
                        idx += 1
                except VisdomExceptionBase:
                    self.create_visdom_connections()
        # 是否使用wandb来作为可视化工具,visdom和wandb二者选择一个就行
        if self.use_wandb:
            columns = [key for key, _ in visuals.items()]
            columns.insert(0, 'epoch')
            result_table = wandb.Table(columns=columns)
            table_row = [epoch]
            ims_dict = {}
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                wandb_image = wandb.Image(image_numpy)
                table_row.append(wandb_image)
                ims_dict[label] = wandb_image
            self.wandb_run.log(ims_dict)
            if epoch != self.current_epoch:
                self.current_epoch = epoch
                result_table.add_data(*table_row)
                self.wandb_run.log({"Result": result_table})

        if self.use_html and (save_result or not self.saved):  # 如果尚未保存图像，请将图像保存到 HTML 文件。
            self.saved = True
            # 将图像保存到磁盘
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)

            # 更新网站
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=0)  # refresh设置为0 不刷新网页
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims, txts, links = [], [], []

                for label, image_numpy in visuals.items():
                    image_numpy = util.tensor2im(image)
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    def plot_current_losses(self, epoch, counter_ratio, losses):
        """在 VISDOM 显示屏上显示当前损失：错误标签和值字典

        Parameters:
            epoch (int)           -- 当前epoch
            counter_ratio (float) -- 当前纪元中的进展（百分比），介于 0 到 1 之间
            losses (OrderedDict)  -- 以 （name， float） 格式存储的训练损失
        """
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
                Y=np.array(self.plot_data['Y']),
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id)
        except VisdomExceptionBase:
            self.create_visdom_connections()
        if self.use_wandb:
            self.wandb_run.log(losses)

    # 损失：与 |loss|plot_current_losses损失的格式相同
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """在控制台上打印损失;还可以将损失保存到磁盘上

        Parameters:
            epoch (int) -- 当前 epoch
            iters (int) -- 此 epoch 期间的当前训练迭代（在每个 epoch 结束时重置为 0）
            losses (OrderedDict) -- 以 （name， float） 对格式存储的训练损失
            t_comp (float) -- 每个数据点的计算时间（按 batch_size 进行归一化）
            t_data (float) -- 每个数据点的数据加载时间（按 batch_size 进行归一化）
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message
