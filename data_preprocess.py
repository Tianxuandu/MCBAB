# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os.path
import copy
from PIL import Image ,ImageEnhance
import random

"""
# 椒盐噪声
def SaltAndPepper(src, percetage):
    SP_NoiseImg = src.copy()
    SP_NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(SP_NoiseNum):
        randR = np.random.randint(0, src.shape[0] - 1)
        randG = np.random.randint(0, src.shape[1] - 1)
        randB = np.random.randint(0, 3)
        if np.random.randint(0, 1) == 0:
            SP_NoiseImg[randR, randG, randB] = 0
        else:
            SP_NoiseImg[randR, randG, randB] = 255
    return SP_NoiseImg


# 高斯噪声
def addGaussianNoise(image, percetage):
    G_Noiseimg = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    G_NoiseNum = int(percetage * image.shape[0] * image.shape[1])
    for i in range(G_NoiseNum):
        temp_x = np.random.randint(0, h)
        temp_y = np.random.randint(0, w)
        G_Noiseimg[temp_x][temp_y][np.random.randint(3)] = np.random.randn(1)[0]
    return G_Noiseimg


# 昏暗
def darker(image, percetage=0.9):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    # get darker
    for xi in range(0, w):
        for xj in range(0, h):
            image_copy[xj, xi, 0] = int(image[xj, xi, 0] * percetage)
            image_copy[xj, xi, 1] = int(image[xj, xi, 1] * percetage)
            image_copy[xj, xi, 2] = int(image[xj, xi, 2] * percetage)
    return image_copy


# 亮度
def brighter(image, percetage=1.5):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    # get brighter
    for xi in range(0, w):
        for xj in range(0, h):
            image_copy[xj, xi, 0] = np.clip(int(image[xj, xi, 0] * percetage), a_max=255, a_min=0)
            image_copy[xj, xi, 1] = np.clip(int(image[xj, xi, 1] * percetage), a_max=255, a_min=0)
            image_copy[xj, xi, 2] = np.clip(int(image[xj, xi, 2] * percetage), a_max=255, a_min=0)
    return image_copy
"""


#旋转
def rotate_image(image_path, angle,expand=True,**kwags):
    image = Image.open(image_path)
    rotated_image = image.rotate(angle, expand=expand)
    return rotated_image

#平移
def translate_image(image_path, x_offset, y_offset,**kwags):
    image = Image.open(image_path)
    width, height = image.size
    # 计算新的图像尺寸，以确保原始图像完全显示
    new_width = width + abs(x_offset)
    new_height = height + abs(y_offset)
    # 创建一个新的图像，背景为黑色
    new_image = Image.new('RGB', (new_width, new_height), (0, 0, 0))
    # 将原始图像粘贴到新图像上
    new_image.paste(image, (x_offset, y_offset))
    return new_image

#错切
def shear_image(image_path, x_shear, y_shear,**kwags):
    image = Image.open(image_path)
    # 计算错切矩阵
    matrix = (1, x_shear, 0, y_shear, 1, 0)
    sheared_image = image.transform(image.size, Image.AFFINE, matrix)
    return sheared_image

#翻转
def flip_image(image_path, flip_type=0,**kwags):
    image = Image.open(image_path)
    if flip_type == 0:
        flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
    elif flip_type == 1:
        flipped_image = image.transpose(Image.FLIP_TOP_BOTTOM)
    return flipped_image


#裁剪和调整大小
def crop_and_resize(image_path, crop_size=(100, 100),**kwargs):
    image = Image.open(image_path).convert('RGB')  # 确保图像是RGB模式
    width, height = image.size

    # 随机选择裁剪的起始点
    if width < crop_size[0] or height < crop_size[1]:
        crop_size = (width//2, height//2)
    start_x = random.randint(0, width - crop_size[0] + 1)
    start_y = random.randint(0, height - crop_size[1] + 1)

    # 裁剪图像
    cropped_image = np.array(image)[start_y:start_y + crop_size[1], start_x:start_x + crop_size[0]]

    # 如果裁剪区域超出了图像的边界，将超出部分用黑色填充
    if cropped_image.shape[0] < crop_size[1] or cropped_image.shape[1] < crop_size[0]:
        padding = [(crop_size[1] - cropped_image.shape[0]) // 2, (crop_size[0] - cropped_image.shape[1]) // 2]
        cropped_image = np.pad(cropped_image, padding, mode='constant', constant_values=0)
    resized_image = Image.fromarray(cropped_image.astype(np.uint8)).resize((width, height))
    # 将NumPy数组转换回PIL图像
    return resized_image




#高斯噪声
def add_gaussian_noise(image_path, mean=5, std=5,**kwags):
    # 打开图片并转换为NumPy数组
    image = Image.open(image_path)
    numpy_image = np.array(image)

    # 确保图像是浮点类型，以便进行数学运算
    if numpy_image.dtype != np.float32:
        numpy_image = numpy_image.astype(np.float32)

    # 生成与图像形状相同的高斯噪声
    gaussian_noise = np.random.normal(mean, std, numpy_image.shape)

    # 将噪声添加到图像上
    noisy_image = np.clip(numpy_image + gaussian_noise, 0, 255).astype(np.uint8)

    # 将NumPy数组转换回PIL图像
    noisy_image_pil = Image.fromarray(noisy_image)

    return noisy_image_pil

#椒盐损失
def add_salt_and_pepper_noise(image_path, noise_ratio,image_type='RGB',**kwags):
    """
    向图像添加椒盐噪声。

    参数:
    - image_path: 图像文件的路径。
    - noise_ratio: 噪声比例，表示噪声点占图像总像素的比例。
    """
    if image_type == 'L':
        # 打开图像并转换为灰度图（如果需要）
        image = Image.open(image_path).convert('L')  # 对于RGB图像，可以去掉.convert('L')来保持颜色
        # 将图像转换为NumPy数组
        image_np = np.array(image)

        # 计算噪声点的数量
        num_noise_pixels = int(noise_ratio * image_np.size)

        # 随机选择噪声点的位置
        coords = [np.random.randint(0, i - 1, num_noise_pixels) for i in image_np.shape]

        # 向这些位置添加椒盐噪声
        image_np[coords[0], coords[1]] = np.random.choice([0, 255], num_noise_pixels)

        # 将NumPy数组转换回图像
        noisy_image = Image.fromarray(image_np)
    if image_type =='RGB':
        image = Image.open(image_path).convert('RGB')
        # 将图像转换为NumPy数组
        image_np = np.array(image)
        width,height = image.size[:2]
        # 计算噪声点的数量
        num_noise_pixels = int(noise_ratio * image_np.size // 3)

        # 为每个颜色通道添加椒盐噪声
        for c in range(3):  # RGB三个通道
            # 随机选择噪声点的位置
            coords = np.random.randint(0, height, num_noise_pixels)
            indices = np.random.randint(0, width, num_noise_pixels)
            # 向这些位置添加椒盐噪声
            image_np[coords, indices, c] = np.random.choice([0, 255], num_noise_pixels)

        # 将NumPy数组转换回图像
        noisy_image = Image.fromarray(image_np.astype('uint8'), 'RGB')
    return noisy_image


#亮度
def brighter(image_path, brightness_factor,**kwags):
    image = Image.open(image_path).convert('RGB')
    # 创建亮度增强器
    enhancer = ImageEnhance.Brightness(image)
    # 调整亮度，factor值大于1增加亮度，小于1减少亮度
    brightened_image = enhancer.enhance(brightness_factor)
    return brightened_image


#对比度
def contrast(image_path, contrast_factor,**kwags):
    image = Image.open(image_path).convert('RGB')
    # 创建对比度增强器
    enhancer = ImageEnhance.Contrast(image)
    # 增加对比度，factor值大于1增加对比度，小于1减少对比度
    enhanced_image = enhancer.enhance(contrast_factor)
    return enhanced_image


#高斯模糊
def apply_gaussian_blur(image_path, kernel_size=(5, 5),**kwags):
    #应用高斯模糊核
    image = Image.open(image_path)
    image = np.array(image)
    return Image.fromarray(cv2.GaussianBlur(image, kernel_size, 0).astype('uint8'))


#Sobel算子边缘检测
def apply_sobel_edge_detection(image_path,**kwags):
    #应用Sobel算子进行边缘检测
    image = np.array(Image.open(image_path))
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    edge_x = cv2.filter2D(image, -1, sobel_x)
    edge_y = cv2.filter2D(image, -1, sobel_y)
    return Image.fromarray(cv2.addWeighted(edge_x, 0.5, edge_y, 0.5, 0).astype('uint8'))


#cutout
def apply_cutout(image_path, max_erosion=100,mix_erosion=100,**kwags):
    #应用Cutout数据增强技术
    # 确定图像尺寸
    image = np.array(Image.open(image_path))
    h, w = image.shape[:2]
    # 随机选择擦除区域的大小
    cutout_area = np.random.randint(1+mix_erosion, max_erosion + 1)
    # 随机选择擦除区域的起始点
    start_x = np.random.randint(0, w)
    start_y = np.random.randint(0, h/2)
    # 计算擦除区域的结束点
    end_x = start_x + cutout_area
    end_y = start_y + cutout_area
    # 确保擦除区域在图像边界内
    end_x = min(end_x, w)
    end_y = min(end_y, h)
    # 应用Cutout，将选定区域的像素值设置为0
    image[start_y:end_y, start_x:end_x] = 0
    return Image.fromarray(image.astype('uint8'))


#random erase
def apply_random_erase(image_path, p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False,convert='RGB',**kwags):
    """
    对图像应用随机擦除增强。
    :param image: 输入图像。
    :param p: 擦除的概率。
    :param s_l: 擦除区域面积的最小比例。
    :param s_h: 擦除区域面积的最大比例。
    :param r_1: 擦除区域宽高比的最小值。
    :param r_2: 擦除区域宽高比的最大值。
    :param v_l: 擦除区域的最小值。
    :param v_h: 擦除区域的最大值。
    :param pixel_level: 是否在像素级别上应用擦除。
    :return: 增强后的图像。
    """
    if np.random.rand() > p:
        return Image.open(image_path).convert(convert)

    # 获取图像尺寸
    image = np.array(Image.open(image_path))
    img_h, img_w, img_c = image.shape

    while True:
        s = np.random.uniform(s_l, s_h) * img_h * img_w
        r = np.random.uniform(r_1, r_2)
        w = int(np.sqrt(s / r))
        h = int(np.sqrt(s * r))
        left = np.random.randint(0, img_w)
        top = np.random.randint(0, img_h/2)

        if left + w <= img_w and top + h <= img_h:
            break

    if pixel_level:
        c = np.random.uniform(v_l, v_h, (h, w, img_c)).astype(image.dtype)
    else:
        c = np.random.uniform(v_l, v_h)

    image[top:top + h, left:left + w, :] = c

    return Image.fromarray(image.astype('uint8'))


#hide and seek
def hide_and_seek(image_path, num_patches, patch_size_range, value=0,**kwags):
    """
    对图像应用Hide-and-Seek数据增强。
    :param image: 输入图像。
    :param num_patches: 要删除的正方形区域的数量。
    :param patch_size_range: 正方形区域大小的范围，格式为(min_size, max_size)。
    :param value: 填充被删除区域的值，默认为0。
    :return: 增强后的图像。
    """
    image = np.array(Image.open(image_path))
    img_h, img_w = image.shape[:2]
    for _ in range(num_patches):
        # 随机选择正方形区域的大小
        patch_size = np.random.randint(patch_size_range[0], patch_size_range[1] + 1)
        # 随机选择正方形区域的起始点
        start_x = np.random.randint(0, img_w - patch_size + 1)
        start_y = np.random.randint(0, img_h - patch_size + 1)
        # 删除正方形区域
        image[start_y:start_y + patch_size, start_x:start_x + patch_size] = value
    return Image.fromarray(image.astype('uint8'))

#GridMask
def generate_grid_mask(image_path, mask_ratio=0.3, cell_size=(10, 10), rotate=False,**kwags):
    img = np.array(Image.open(image_path).convert('RGB'))
    h, w, c = img.shape
    mask = np.ones((h, w, c), dtype=np.uint8)  # 确保掩码是三个通道
    w_cell, h_cell = cell_size
    num_cells_x = w // w_cell
    num_cells_y = h // h_cell
    num_cells_to_mask = int(num_cells_x * num_cells_y * mask_ratio)
    cells_to_mask = np.random.choice(range(num_cells_x * num_cells_y), num_cells_to_mask, replace=False)
    for i in cells_to_mask:
        x_start = i % num_cells_x * w_cell
        y_start = i // num_cells_x * h_cell
        # 应用掩码到三个通道
        for ch in range(c):  # 对每个通道进行操作
            mask[y_start:y_start + h_cell, x_start:x_start + w_cell, ch] = 255
    masked_img = cv2.bitwise_and(img, mask)
    if rotate:
        angle = np.random.choice([90, 180, 270], 1)[0]
        masked_img = cv2.rotate(masked_img, cv2.ROTATE_90_CLOCKWISE * angle)
    return Image.fromarray(masked_img.astype('uint8'))


"""
image='C:/Users/Dumin/Desktop/fishnet/dataset/000c42ab-e5ff-433b-9d62-4ba4126f5cb0.jpg'
rotated_img = add_gaussian_noise('C:/Users/Dumin/Desktop/fishnet/dataset/000c42ab-e5ff-433b-9d62-4ba4126f5cb0.jpg')
print(rotated_img.size)# 旋转45度
rotated_img.show()

"""



















