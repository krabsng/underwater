import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from uiqe import uiqe
from uiqm import uiqm


def calculate_metrics(image1, image2):
    # 计算PSNR
    psnr_value = psnr(image1, image2)

    # 计算SSIM
    ssim_value, _ = ssim(image1, image2, full=True)

    # 计算UIQM
    uiqm_value = uiqm(image1)

    # 计算UIQE
    uiqe_value = uiqe(image1)

    return psnr_value, ssim_value, uiqm_value, uiqe_value


def calculate_average_metrics(image_list1, image_list2):
    psnr_values = []
    ssim_values = []
    uiqm_values = []
    uiqe_values = []

    # 对每对图片计算各个指标
    for img1_path, img2_path in zip(image_list1, image_list2):
        # 读取图片
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        # 将图片转换为灰度图（对于SSIM和PSNR，通常使用灰度图）
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # 计算各个指标
        psnr_val, ssim_val, uiqm_val, uiqe_val = calculate_metrics(img1_gray, img2_gray)

        psnr_values.append(psnr_val)
        ssim_values.append(ssim_val)
        uiqm_values.append(uiqm_val)
        uiqe_values.append(uiqe_val)

    # 计算平均值
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    avg_uiqm = np.mean(uiqm_values)
    avg_uiqe = np.mean(uiqe_values)

    return avg_psnr, avg_ssim, avg_uiqm, avg_uiqe


if __name__ == "__main__":
    # 图片路径列表
    image_list1 = ["image1.png", "image2.png", "image3.png"]  # 原图列表
    image_list2 = ["image1_pred.png", "image2_pred.png", "image3_pred.png"]  # 重建/预测图列表

    avg_psnr, avg_ssim, avg_uiqm, avg_uiqe = calculate_average_metrics(image_list1, image_list2)

    print(f"Average PSNR: {avg_psnr:.4f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average UIQM: {avg_uiqm:.4f}")
    print(f"Average UIQE: {avg_uiqe:.4f}")
