"""
    傅立叶变换去噪算法，傅里叶变换常用于频域滤波，来去除图像中的高频或低频噪声
"""
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def high_pass_filter_channel(channel: np.ndarray, cutoff: int = 30) -> np.ndarray:
    """
    对单个通道应用高通滤波器，使用傅里叶变换去除低频噪声。

    参数:
        channel: 输入图像的单个通道，NumPy数组格式。
        cutoff: 高通滤波器的截止频率。

    返回:
        经过高通滤波后的图像通道。
    """
    # 图像的傅里叶变换
    f = np.fft.fft2(channel)
    fshift = np.fft.fftshift(f)

    # 创建高通滤波器的掩模
    rows, cols = channel.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    mask[crow - cutoff:crow + cutoff, ccol - cutoff:ccol + cutoff] = 0

    # 应用掩模并进行逆傅里叶变换
    fshift = fshift * mask
    f_ishift = np.fft.ifftshift(fshift)
    channel_back = np.fft.ifft2(f_ishift)
    channel_back = np.abs(channel_back)

    return channel_back


def high_pass_filter_color_image(img: np.ndarray, cutoff: int = 30) -> np.ndarray:
    """
    对彩色图像应用高通滤波器，使用傅里叶变换去除低频噪声。

    参数:
        img: 输入图像，RGB格式的NumPy数组。
        cutoff: 高通滤波器的截止频率。

    返回:
        经过高通滤波后的彩色图像。
    """
    # 分离通道
    r, g, b = cv2.split(img)

    # 对每个通道应用高通滤波
    r_filtered = high_pass_filter_channel(r, cutoff)
    g_filtered = high_pass_filter_channel(g, cutoff)
    b_filtered = high_pass_filter_channel(b, cutoff)

    # 合并通道
    filtered_img = cv2.merge([r_filtered, g_filtered, b_filtered])

    # 确保像素值在0-255范围内
    filtered_img = np.clip(filtered_img, 0, 255).astype(np.uint8)

    return filtered_img


if __name__ == "__main__":
    # 打开图片并转换为RGB格式的NumPy数组
    img = Image.open("/home/ljp/a.krabs/waternet-main/docs/frames-0968-small.jpeg").convert("RGB")
    img = np.array(img)

    # 应用高通滤波器去除低频噪声
    cutoff = 30  # 选择一个合适的截止频率
    denoised_img = high_pass_filter_color_image(img, cutoff)

    # 显示去噪前后的图像
    plt.subplot(121), plt.imshow(img)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(denoised_img)
    plt.title('High Pass Filtered Image'), plt.xticks([]), plt.yticks([])
    plt.show()
