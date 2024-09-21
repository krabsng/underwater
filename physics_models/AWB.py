"""
    动态白平衡算法
"""
import cv2
import numpy as np


def automatic_white_balance(image):
    # 分离BGR通道
    B, G, R = cv2.split(image)

    # 计算每个通道的平均值
    B_avg = np.mean(B)
    G_avg = np.mean(G)
    R_avg = np.mean(R)

    # 计算灰度世界的增益
    K = (B_avg + G_avg + R_avg) / 3
    B_gain = K / B_avg
    G_gain = K / G_avg
    R_gain = K / R_avg

    # 调整每个通道
    B = cv2.convertScaleAbs(B * B_gain)
    G = cv2.convertScaleAbs(G * G_gain)
    R = cv2.convertScaleAbs(R * R_gain)

    # 合并通道
    balanced_image = cv2.merge([B, G, R])

    return balanced_image


if __name__ == "__main__":
    input_image_path = "/home/ljp/a.krabs/waternet-main/docs/frames-0968-small.jpeg"  # 替换为实际的图像路径
    output_image_path = "awb_color_image.jpg"

    # 读取图像
    image = cv2.imread(input_image_path)

    # 应用自动白平衡算法
    awb_image = automatic_white_balance(image)

    # 保存并显示结果
    cv2.imwrite(output_image_path, awb_image)
    cv2.imshow('AWB Color Image', awb_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
