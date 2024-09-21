"""
    SIFT特征点匹配算法
    使用K-近邻（KNN）算法在两幅图像之间匹配特征点，并应用比率测试来筛选匹配
"""
import cv2
import numpy as np

def sift_feature_matching(image1_path, image2_path):
    # 读取彩色图像
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    # 转换为灰度图像
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 初始化SIFT检测器
    sift = cv2.SIFT_create()

    # 检测关键点并计算描述符
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # 使用FLANN匹配器进行特征匹配
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # 应用比率测试
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # 画出匹配的关键点（匹配图像中的特征点）
    result_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return result_img


if __name__ == "__main__":
    image1_path = "/home/ljp/a.krabs/dataset/UIQS(RUIE)/UIQS/pic_A/JPEGImages/3200.jpg"  # 替换为实际的图像路径
    image2_path = "/home/ljp/a.krabs/dataset/UIQS(RUIE)/UIQS/pic_A/JPEGImages/3195.jpg"  # 替换为实际的图像路径
    output_image_path = "matched_image.jpg"

    matched_image = sift_feature_matching(image1_path, image2_path)

    # 保存并显示结果
    cv2.imwrite(output_image_path, matched_image)
    cv2.imshow('Matched Image', matched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
