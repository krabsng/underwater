"""Canny边缘检测
"""
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def get_edge_img(img_dir:str):
    image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    # 应用高斯滤波器进行降噪
    """
        parameters：
            Ksize（5，5） --高斯核的大小，用于定义卷积核的大小。
            sigmaX 1。4  --X方向的高斯核标准差。这个参数决定了模糊的程度。值越大，模糊越强
    """
    blurred_image = cv2.GaussianBlur(image, (5, 5), 1.4)
    # 使用Canny边缘检测
    edges = cv2.Canny(blurred_image, 50, 150)
    return image, edges


    plt.show()

# 测试该函数好不好用
if __name__ == '__main__':
    image, edges = get_edge_img('/home/ljp/a.krabs/krabs/results/iat/test_latest/images/test_p0__GT_Img.png')
    # 显示原图和边缘检测结果
    plt.figure(figsize=(10, 5))

    plt.subplot(121)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(edges, cmap='gray')
    plt.title('Canny Edges')
    plt.axis('off')
    image_pil = Image.fromarray(edges)
    image_pil.save("/home/ljp/a.krabs/krabs/evaluation/edges.jpg")