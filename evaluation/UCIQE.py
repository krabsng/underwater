import cv2
import numpy as np
import math

def calculate_uciqe(image):
    # 将图像从 BGR 转换为 HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # 计算色度的标准差（delta）
    delta = np.std(H) / 180.0

    # 计算饱和度的平均值（mu）
    mu = np.mean(S) / 255.0

    # 计算亮度（V）的对比度（conl）
    n, m = V.shape
    number = math.floor(n * m / 100)
    V1 = V.flatten() / 255.0
    V_sorted = np.sort(V1)

    top = np.mean(V_sorted[-number:])
    bottom = np.mean(V_sorted[:number])
    conl = top - bottom

    # 计算 UCIQE 值
    uciqe = 0.4680 * delta + 0.2745 * conl + 0.2576 * mu

    return uciqe


if __name__=='__main__':
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    image1 = cv2.imread("./test_p98__GT_Img.png")
    image2 = cv2.imread("./test_p98__Generate_Img.png")
    uciqe_value = calculate_uciqe(image1)
    ssim_value = ssim(image2,image1,full=True,win_size=3)[0]
    psnr_value = psnr(image2,image1,)
    print("UCIQE:", uciqe_value)
    print("SSIM:", ssim_value)
    print("PSNR:", psnr_value)


