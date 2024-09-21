"""
    直方图均衡化算法
"""
import cv2

"""
    clip_limit: 这是CLAHE算法中的对比度限制参数，通常设为2.0。值越大，对比度增强越明显。
    tile_grid_size: 表示图像被划分为多少个网格（如8x8），每个网格应用一次自适应直方图均衡化。
"""
def apply_clahe_to_color_image(image, clip_limit=2.0, tile_grid_size=(8, 8)):


    # 转换为Lab颜色空间
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    # 拆分Lab图像到L, a, b通道
    l_channel, a_channel, b_channel = cv2.split(lab)

    # 创建CLAHE对象
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    # 对L通道应用CLAHE
    l_channel = clahe.apply(l_channel)

    # 合并增强后的L通道和原始的a, b通道
    enhanced_lab = cv2.merge((l_channel, a_channel, b_channel))

    # 将Lab图像转换回BGR颜色空间
    enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_Lab2BGR)

    return enhanced_image

if __name__ == "__main__":
    input_image_path = "/home/ljp/a.krabs/krabs/tmp/frame_0000.jpg"  # 替换为实际图像路径
    output_image_path = "enhanced_color_image.jpg"

    # 读取彩色图像
    image = cv2.imread(input_image_path)

    # 应用CLAHE算法处理彩色图像
    enhanced_image = apply_clahe_to_color_image(image)

    # 保存并显示结果
    cv2.imwrite(output_image_path, enhanced_image)
    cv2.imshow('Enhanced Color Image', enhanced_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
