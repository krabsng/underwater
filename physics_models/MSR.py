"""
    多尺度Retinex算法
"""
import cv2
import numpy as np

def single_scale_retinex(img, sigma):
    # 单尺度Retinex
    retinex = np.log10(img + 1.0) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma) + 1.0)
    return retinex

def multi_scale_retinex(img, sigma_list):
    # 多尺度Retinex
    retinex = np.zeros_like(img)
    for sigma in sigma_list:
        retinex += single_scale_retinex(img, sigma)
    retinex = retinex / len(sigma_list)
    return retinex

def msr_algorithm(image_path):
    # 读取图像
    img = cv2.imread(image_path)
    img = img.astype(np.float64) / 255.0

    # 分通道处理
    img_msr = np.zeros_like(img)
    for i in range(3):
        img_msr[:, :, i] = multi_scale_retinex(img[:, :, i], sigma_list=[15, 80, 250])

    # 对比度拉伸
    for i in range(3):
        img_msr[:, :, i] = np.clip((img_msr[:, :, i] - np.min(img_msr[:, :, i])) /
                                   (np.max(img_msr[:, :, i]) - np.min(img_msr[:, :, i])) * 255, 0, 255)

    img_msr = img_msr.astype(np.uint8)

    return img_msr

if __name__ == "__main__":
    input_image_path = "/home/ljp/a.krabs/krabs/tmp/frame_0000.jpg"  # 替换为实际图像路径
    output_image_path = "enhanced_msr_image.jpg"

    enhanced_image = msr_algorithm(input_image_path)
    cv2.imwrite(output_image_path, enhanced_image)
    cv2.imshow('Enhanced MSR Image', enhanced_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
