"""
    多尺度融合水下图像增强算法
"""
import cv2
import numpy as np


def apply_clahe(channel):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(channel)


def enhance_image(image):
    # 将图像转换为LAB颜色空间
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    # 对L通道应用CLAHE（限制对比度自适应直方图均衡化）
    l, a, b = cv2.split(lab)
    l = apply_clahe(l)

    # 合并增强后的L通道
    enhanced_lab = cv2.merge((l, a, b))

    # 将图像转换回BGR颜色空间
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_Lab2BGR)

    return enhanced_bgr


def multi_scale_fusion(image):
    # 创建两个不同的版本的图像
    image1 = enhance_image(image)  # 基于CLAHE的增强图像
    image2 = cv2.detailEnhance(image, sigma_s=10, sigma_r=0.15)  # 细节增强图像

    # 融合两个图像
    fused_image = cv2.addWeighted(image1, 0.5, image2, 0.5, 0)

    return fused_image


if __name__ == "__main__":
    input_image_path = "/home/ljp/a.krabs/krabs/tmp/frame_0000.jpg"  # 替换为实际图像路径
    output_image_path = "fused_image.jpg"

    # 读取图像
    image = cv2.imread(input_image_path)

    # 应用多尺度融合增强算法
    enhanced_image = multi_scale_fusion(image)

    # 保存并显示结果
    cv2.imwrite(output_image_path, enhanced_image)
    cv2.imshow('Enhanced Image', enhanced_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
