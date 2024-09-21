"""
    gamma矫正算法
"""
import cv2
import numpy as np
from PIL import Image


def gamma_correction(img: np.ndarray, gamma: float) -> np.ndarray:
    """
    对输入的RGB图像应用伽马校正。

    参数:
        img: 输入图像，必须是RGB格式的NumPy数组。
        gamma: 伽马值，大于1会使图像变暗，小于1会使图像变亮。

    返回:
        经过伽马校正后的图像。
    """
    # 建立一个查找表，适用于所有像素值
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in range(256)]).astype("uint8")

    # 使用查找表来对图像进行映射
    corrected_img = cv2.LUT(img, table)

    return corrected_img


if __name__ == "__main__":
    # 打开图片并转换为RGB格式的NumPy数组
    img = Image.open("/home/ljp/a.krabs/waternet-main/docs/frames-0968-small.jpeg")
    img = np.array(img)

    # 应用伽马校正算法
    gamma_value = 2.2  # 选择一个合适的伽马值
    corrected_img = gamma_correction(img, gamma_value)

    # 转换为PIL图像并显示
    corrected_img = Image.fromarray(corrected_img)
    corrected_img.show()
