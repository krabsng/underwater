"""
    ORB特征点匹配算法
    使用K-近邻（KNN）算法在两幅图像之间匹配特征点，并应用比率测试来筛选匹配
"""
import cv2
import numpy as np

def orb_feature_matching(image1_path, image2_path):
    # 读取彩色图像
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    # 转换为灰度图像
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 初始化ORB检测器
    orb = cv2.ORB_create()

    # 检测关键点并计算描述符
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # 使用BFMatcher进行特征匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # 按照匹配的距离排序，距离越小越好
    matches = sorted(matches, key=lambda x: x.distance)

    # 画出匹配的关键点
    result_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return result_img

if __name__ == "__main__":
    image1_path = "/home/ljp/a.krabs/dataset/UIQS(RUIE)/UIQS/pic_A/JPEGImages/3200.jpg"  # 替换为实际的图像路径
    image2_path = "/home/ljp/a.krabs/dataset/UIQS(RUIE)/UIQS/pic_A/JPEGImages/3195.jpg"  # 替换为实际的图像路径
    output_image_path = "matched_orb_image.jpg"

    matched_image = orb_feature_matching(image1_path, image2_path)

    # 保存并显示结果
    cv2.imwrite(output_image_path, matched_image)
    cv2.imshow('Matched ORB Image', matched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
