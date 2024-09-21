"""此模块包含简单的辅助函数 """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import glob
import random
import mimetypes
import re
import cv2
def tensor2im(input_image, imtype=np.uint8):
    """"将 Tensor 数组转换为 numpy 图像数组。

    Parameters:
        input_image (tensor) --  输入图像张量数组,是batch_size个图片
        imtype (type)        --  转换后的 numPy 数组的所需类型
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # 从变量中获取数据
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # 把图像转换为numpy数组
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0 #这里进行了反标准化  # post-processing: tranpose and scaling
    else:  # 如果已经是numpy数组,不做处理
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """计算并打印绝对（梯度）的平均值

    Parameters:
        net (torch network) -- Torch 网络
        name (str) -- 网络的名称
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """将 numpy 图像保存到磁盘

    Parameters:
        image_numpy (numpy array) -- 输入的 numpy 数组
        image_path (str)          -- 图像的路径
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """打印 numpy 数组的平均值、最小值、最大值、中值、std 和大小

    Parameters:
        val (bool) -- 是否打印 numpy 数组的值
        shp (bool) -- 是否打印 numpy 数组的形状
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """如果指定名称的录不存在，请创建空目录

    Parameters:
        paths (str list) -- 目录路径列表
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """创建一个空目录（如果该目录不存在）

    Parameters:
        path (str) -- 单个目录路径
    """
    if not os.path.exists(path):
        os.makedirs(path)

def populate_train_list(images_path, mode='train', pattern='*/trainA/*.jpg'):
    # print(images_path)
    # glob模块，用来查找符合通配好符的文件
    jpeg_pattern = pattern.replace('jpg', 'JPEG')
    image_list_lowlight = glob.glob(images_path + pattern, recursive=True) + glob.glob(images_path + jpeg_pattern, recursive=True)
    train_list = image_list_lowlight
    if mode == 'train':
        random.shuffle(train_list)
    return train_list



def classify_input(input_str):
    """判断输入的字符串的类型（图片、视频、文件夹、URL地址）

    parameters：
        input_str --输入的字符串
    return:
        str --【URL、Folder、Image、Video、File、Unknown】其中一种
    """
    # Regular expression for matching a URL
    url_pattern = re.compile(
        r'^(https?|ftp)://'  # http:// or https:// or ftp://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|'  # ...or ipv4
        r'\[?[A-F0-9]*:[A-F0-9:]+\]?)'  # ...or ipv6
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)

    if re.match(url_pattern, input_str):
        return "URL"

    # Check if it's a valid file path
    if os.path.exists(input_str):
        if os.path.isdir(input_str):
            return "Folder"
        else:
            # Guess the type of the file
            mime_type, _ = mimetypes.guess_type(input_str)
            if mime_type:
                if mime_type.startswith('image'):
                    return "Image"
                elif mime_type.startswith('video'):
                    return "Video"
            return "File"

    return "Unknown"

def video_to_frames(video_path, output_dir):
    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 初始化帧计数器
    frame_count = 0

    while True:
        # 逐帧读取视频
        ret, frame = cap.read()

        # 如果没有更多帧，退出循环
        if not ret:
            break

        # 构建图像文件的输出路径
        output_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")

        # 保存帧为图像文件
        cv2.imwrite(output_path, frame)

        # 打印保存信息
        print(f"Saved {output_path}")

        # 增加帧计数器
        frame_count += 1

    # 释放视频捕获对象
    cap.release()