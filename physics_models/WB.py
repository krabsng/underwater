"""
    白平衡算法
"""
import cv2
import numpy as np
from PIL import Image


def white_balance(img: np.ndarray) -> np.ndarray:
    """
    应用白平衡到输入的RGB图像。

    参数:
        img: 输入图像，必须是RGB格式的NumPy数组。

    返回:
        应用白平衡后的图像。
    """
    # 分离BGR通道
    r, g, b = cv2.split(img)

    # 计算每个通道的均值
    r_avg = np.mean(r)
    g_avg = np.mean(g)
    b_avg = np.mean(b)

    # 计算整个图像的灰度均值
    gray_avg = (r_avg + g_avg + b_avg) / 3

    # 计算每个通道的增益因子
    r_gain = gray_avg / r_avg
    g_gain = gray_avg / g_avg
    b_gain = gray_avg / b_avg

    # 应用增益因子
    r = cv2.multiply(r, r_gain)
    g = cv2.multiply(g, g_gain)
    b = cv2.multiply(b, b_gain)

    # 合并通道
    balanced_img = cv2.merge([r, g, b])

    # 确保输出图像的范围在0-255之间
    balanced_img = np.clip(balanced_img, 0, 255).astype(np.uint8)

    return balanced_img


if __name__ == "__main__":
    # 打开图片并转换为RGB格式的NumPy数组
    img = Image.open("/home/ljp/a.krabs/waternet-main/docs/frames-0968-small.jpeg")
    img = np.array(img)

    # 应用白平衡算法
    balanced_img = white_balance(img)

    # 转换为PIL图像并显示
    balanced_img = Image.fromarray(balanced_img)
    balanced_img.show()
