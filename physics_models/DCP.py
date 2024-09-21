"""
    暗通道先验法
"""
import cv2
import numpy as np

"""
    计算图像的暗通到
"""
def get_dark_channel(image, window_size):
    min_channel = np.min(image, axis=2)
    dark_channel = cv2.erode(min_channel, np.ones((window_size, window_size)))
    return dark_channel

"""
    估计大气光
"""
def get_atmospheric_light(image, dark_channel):
    h, w = image.shape[:2]
    num_pixels = h * w
    num_brightest = int(max(num_pixels * 0.001, 1))

    flat_image = image.reshape(num_pixels, 3)
    flat_dark = dark_channel.ravel()

    indices = np.argsort(flat_dark)[-num_brightest:]
    atmospheric_light = np.mean(flat_image[indices], axis=0)
    return atmospheric_light

"""
    计算图像的传输图，描述光的投射率
"""
def get_transmission(image, atmospheric_light, omega, window_size):
    norm_image = image / atmospheric_light
    dark_channel = get_dark_channel(norm_image, window_size)
    transmission = 1 - omega * dark_channel
    return transmission

"""
    使用逆向物理模型恢复图像的颜色和对比度
"""
def recover_image(image, transmission, atmospheric_light, t0):
    transmission = np.maximum(transmission, t0)
    recovered = (image - atmospheric_light) / transmission[:, :, np.newaxis] + atmospheric_light
    return np.clip(recovered, 0, 255).astype(np.uint8)

def dark_channel_prior(image_path):
    image = cv2.imread(image_path)
    image = image.astype(np.float64)

    window_size = 15
    omega = 0.95
    t0 = 0.1

    dark_channel = get_dark_channel(image, window_size)
    atmospheric_light = get_atmospheric_light(image, dark_channel)
    transmission = get_transmission(image, atmospheric_light, omega, window_size)
    recovered_image = recover_image(image, transmission, atmospheric_light, t0)

    return recovered_image

if __name__ == "__main__":
    input_image_path = "/home/ljp/a.krabs/krabs/tmp/frame_0000.jpg"  # 替换为你实际的图像路径
    output_image_path = "enhanced_image.jpg"

    enhanced_image = dark_channel_prior(input_image_path)
    cv2.imwrite(output_image_path, enhanced_image)
    cv2.imshow('Enhanced Image', enhanced_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
