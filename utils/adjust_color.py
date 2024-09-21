from PIL import Image

def adjust_rgb(image_path, output_path, red_factor=1.0, green_factor=1.0, blue_factor=1.0):
    # 打开图像
    img = Image.open(image_path)

    # 将图像转换为RGB模式（如果不是的话）
    img = img.convert("RGB")

    # 获取像素数据
    pixels = img.load()

    # 遍历所有像素
    for i in range(img.width):
        for j in range(img.height):
            r, g, b = pixels[i, j]

            # 调整RGB通道值
            r = int(r * red_factor)
            g = int(g * green_factor)
            b = int(b * blue_factor)

            # 防止溢出到[0, 255]以外
            r = max(0, min(255, r))
            g = max(0, min(255, g))
            b = max(0, min(255, b))

            # 将新像素值设置回去
            pixels[i, j] = (r, g, b)

    # 保存调整后的图像
    img.save(output_path)
    print(f"图像已保存到 {output_path}")