"""
    水下暗通道先验法
"""
import cv2
import numpy as np


def get_udcp_dark_channel(image, window_size):
    # 对每个颜色通道分别计算暗通道
    b, g, r = cv2.split(image)

    # 使用蓝色通道（通常是水下图像中受影响最大的通道）作为计算基础
    min_channel = np.minimum(b, np.minimum(g, r))
    dark_channel = cv2.erode(min_channel, np.ones((window_size, window_size), np.uint8))

    return dark_channel


def get_atmospheric_light_udcp(image, dark_channel):
    # 基于暗通道获取大气光
    h, w = image.shape[:2]
    num_pixels = h * w
    num_brightest = int(max(num_pixels * 0.001, 1))

    flat_image = image.reshape(num_pixels, 3)
    flat_dark = dark_channel.ravel()

    indices = np.argsort(flat_dark)[-num_brightest:]
    atmospheric_light = np.mean(flat_image[indices], axis=0)

    return atmospheric_light


def get_transmission_udcp(image, atmospheric_light, omega, window_size):
    # 计算透射率
    norm_image = image / atmospheric_light
    dark_channel = get_udcp_dark_channel(norm_image, window_size)
    transmission = 1 - omega * dark_channel

    return transmission


def recover_image_udcp(image, transmission, atmospheric_light, t0):
    # 恢复图像
    transmission = np.maximum(transmission, t0)
    recovered = (image - atmospheric_light) / transmission[:, :, np.newaxis] + atmospheric_light
    return np.clip(recovered, 0, 255).astype(np.uint8)


def udcp_algorithm(image_path):
    image = cv2.imread(image_path)
    image = image.astype(np.float64)

    window_size = 15
    omega = 0.95
    t0 = 0.1

    dark_channel = get_udcp_dark_channel(image, window_size)
    atmospheric_light = get_atmospheric_light_udcp(image, dark_channel)
    transmission = get_transmission_udcp(image, atmospheric_light, omega, window_size)
    recovered_image = recover_image_udcp(image, transmission, atmospheric_light, t0)

    return recovered_image


if __name__ == "__main__":
    input_image_path = "/home/ljp/a.krabs/krabs/tmp/frame_0000.jpg"  # 替换为实际的图像路径
    output_image_path = "enhanced_udcp_image.jpg"

    enhanced_image = udcp_algorithm(input_image_path)
    cv2.imwrite(output_image_path, enhanced_image)
    cv2.imshow('Enhanced UDCP Image', enhanced_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
