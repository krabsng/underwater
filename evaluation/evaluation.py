import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from UCIQE import calculate_uciqe as uciqe
from UIQM import compute_uiqm as uiqm


def calculate_metrics(image1, image2):
    # 计算PSNR
    psnr_value = psnr(image1, image2)

    # 计算SSIM
    ssim_value, _ = ssim(image1, image2, win_size=3,full=True)

    # 计算UIQM
    uiqm_value = uiqm(image1)

    # 计算UCIQE
    uciqe_value = uciqe(image1)

    return psnr_value, ssim_value, uiqm_value, uciqe_value


def calculate_average_metrics(image_list1, image_list2):
    psnr_values = []
    ssim_values = []
    uiqm_values = []
    uciqe_values = []

    # 对每对图片计算各个指标
    for img1_path, img2_path in zip(image_list1, image_list2):
        # 读取图片
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        # # 将图片转换为灰度图（对于SSIM和PSNR，通常使用灰度图）
        # img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        # img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # 计算各个指标
        psnr_val, ssim_val, uiqm_val, uiqe_val = calculate_metrics(img1, img2)

        psnr_values.append(psnr_val)
        ssim_values.append(ssim_val)
        uiqm_values.append(uiqm_val)
        uciqe_values.append(uiqe_val)

    # 计算平均值
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    avg_uiqm = np.mean(uiqm_values)
    avg_uciqe = np.mean(uciqe_values)

    return avg_psnr, avg_ssim, avg_uiqm, avg_uciqe


def split_images_by_keywords(folder_path, keyword1, keyword2):
    """
    遍历指定文件夹，搜索图片文件并根据文件名中是否包含关键字将文件分为两个组：
    - 第一组：包含关键字1的图片
    - 第二组：同时包含关键字1和关键字2的图片

    :param folder_path: 文件夹路径
    :param keyword1: 第一关键字，用于分类
    :param keyword2: 第二关键字，用于第二组分类
    :return: 返回两个列表：
            - 包含关键字1的文件路径列表
            - 同时包含关键字1和关键字2的文件路径列表
    """
    # 用于存放分组的两个列表
    group1 = []
    group2 = []

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查是否是图片文件，文件扩展名为 .jpg、.jpeg 或 .png（可以根据需要添加其他格式）
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            # 获取文件的绝对路径
            file_path = os.path.join(folder_path, filename)

            # 判断是否包含第一个关键字
            if keyword1.lower() in filename.lower():
                group1.append(file_path)
            # 包含第二个关键字,加入第二组
            elif keyword2.lower() in filename.lower():
                group2.append(file_path)

    return group1, group2


if __name__ == "__main__":
    # 图片路径列表
    image_list1,image_list2 = split_images_by_keywords('/a.krabs/krabs/results/iat/test_latest','Generate','GT')  # 原图列表
    print(image_list1)
    print(image_list2)
    avg_psnr, avg_ssim, avg_uiqm, avg_uciqe = calculate_average_metrics(image_list1, image_list2)

    print(f"Average PSNR: {avg_psnr:.4f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average UIQM: {avg_uiqm:.4f}")
    print(f"Average UCIQE: {avg_uciqe:.4f}")
