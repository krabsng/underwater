import numpy as np
import cv2

def compute_uicm(image):
    # 色彩失真 (UICM)
    R, G, B = image[:,:,2], image[:,:,1], image[:,:,0]
    rg = R - G
    yb = 0.5 * (R + G) - B
    rg_mean = np.mean(rg)
    yb_mean = np.mean(yb)
    rg_std = np.std(rg)
    yb_std = np.std(yb)
    uicm = -(rg_std + yb_std) - 0.026 * (rg_mean**2 + yb_mean**2)
    return uicm

def compute_uism(image):
    # 亮度对比度 (UISM)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gradient = np.gradient(gray_image.astype("float"))
    uism = np.std(gradient)
    return uism

def compute_uiconm(image):
    # 清晰度 (UIConM)
    R, G, B = image[:,:,2], image[:,:,1], image[:,:,0]
    r_contrast = np.std(R) / np.mean(R)
    g_contrast = np.std(G) / np.mean(G)
    b_contrast = np.std(B) / np.mean(B)
    uiconm = (r_contrast + g_contrast + b_contrast) / 3
    return uiconm

def compute_uiqm(image):
    # 计算综合评价指标 UIQM
    uicm = compute_uicm(image)
    uism = compute_uism(image)
    uiconm = compute_uiconm(image)
    uiqm = 0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm
    return uiqm

if __name__ == '__main__':
    image = cv2.imread("./img.png")
    uiqm_value = compute_uiqm(image)
    print("UIQM:", uiqm_value)
