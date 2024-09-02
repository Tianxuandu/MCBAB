import pandas as pd
import numpy as np
import os
import data_preprocess
import cv2
import random
from PIL import Image

df_train = pd.read_csv('C:/Users/Dumin/Desktop/cv4e_train.csv')

# 定义图像路径
all_old_img_path = 'C:/Users/Dumin/Desktop/fishnet/dataset'
all_new_img_path = 'C:/Users/Dumin/Desktop/fishnet/fishnet_data_enhance'
all_img_path_train = os.path.join(all_new_img_path, 'train')

category_counts = df_train.iloc[:, 2].value_counts()  # 假设第三列是类别标签
print(category_counts)
def ensure_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
        return path
    except TypeError:
        print(f"Error: The path '{path}' is not valid.")
        return None


def data_enhance(old_img_path, new_img_path,category):
    # 定义数据增强函数列表
    if category == 'One':
        data_enhance_functions = [
            data_preprocess.generate_grid_mask,
            data_preprocess.brighter,
            data_preprocess.crop_and_resize,
            data_preprocess.add_salt_and_pepper_noise,
            data_preprocess.contrast,
            data_preprocess.rotate_image,
            data_preprocess.apply_gaussian_blur,
            data_preprocess.apply_sobel_edge_detection,
            data_preprocess.crop_and_resize,
        ]
    elif category == 'Two':
        data_enhance_functions = [
            data_preprocess.generate_grid_mask,
            data_preprocess.crop_and_resize,
            data_preprocess.apply_gaussian_blur,
        ]
    elif category == 'Three':
        data_enhance_functions = [
            data_preprocess.generate_grid_mask,
            data_preprocess.brighter,
            data_preprocess.crop_and_resize,
            data_preprocess.add_salt_and_pepper_noise,
            data_preprocess.contrast,
            data_preprocess.rotate_image,
            data_preprocess.apply_gaussian_blur,
            data_preprocess.apply_sobel_edge_detection,

            data_preprocess.crop_and_resize,
            data_preprocess.rotate_image,
            data_preprocess.generate_grid_mask,

            data_preprocess.crop_and_resize,
            data_preprocess.rotate_image,
            data_preprocess.generate_grid_mask,

            data_preprocess.crop_and_resize,
            data_preprocess.rotate_image,
            data_preprocess.generate_grid_mask,

            data_preprocess.crop_and_resize,
            data_preprocess.crop_and_resize,
            data_preprocess.crop_and_resize,
            data_preprocess.crop_and_resize,
            data_preprocess.crop_and_resize,
            data_preprocess.crop_and_resize,
            data_preprocess.crop_and_resize,
            data_preprocess.crop_and_resize,
            data_preprocess.crop_and_resize,
            data_preprocess.crop_and_resize,
            data_preprocess.crop_and_resize,
            data_preprocess.crop_and_resize,

        ]
    # 为old_img_path创建Image对象
    image = Image.open(old_img_path)

    for function in data_enhance_functions:
        # 调用函数并传入相应的参数
        enhanced_image = function(image_path=old_img_path, **{'brightness_factor': random.uniform(0.5, 1.5) if function == data_preprocess.brighter else None,
                                                              'crop_size':(300,300) if function == data_preprocess.crop_and_resize else None,
                                                              'contrast_factor':random.uniform(0.5,1.5) if function == data_preprocess.contrast else None,
                                                              'angle':random.uniform(-90,90) if function == data_preprocess.rotate_image else None,
                                                              'noise_ratio':random.uniform(0.1,0.3) if function == data_preprocess.add_salt_and_pepper_noise else None,})
        # 构造新的文件名，基于原文件名和增强函数的名称
        base_name = os.path.basename(old_img_path)
        new_file_name = f"{os.path.splitext(base_name)[0]}_{function.__name__}{os.path.splitext(base_name)[1]}"
        # 保存增强后的图像到new_img_path目录
        enhanced_image.save(os.path.join(new_img_path, new_file_name))


for i in range(len(df_train)):
    img_train = df_train.iloc[i, 0]
    train_family_name = df_train.iloc[i, 3]
    train_genus_name = df_train.iloc[i, 1]
    train_species_name = df_train.iloc[i, 2]
    train_name = os.path.join(all_img_path_train, train_family_name, train_genus_name, train_species_name)
    new_path = ensure_dir(train_name)
    if new_path is None:
        print(f"Error: The directory '{train_name}' could not be created or accessed.")
    else:
        cv2.imwrite(os.path.join(new_path, img_train), cv2.imread(os.path.join(all_old_img_path, img_train)))
        if category_counts[train_species_name] <= 50:
            if category_counts[train_species_name]>=6 and category_counts[train_species_name]<=20:
                data_enhance(os.path.join(all_old_img_path, img_train), new_path,category='One')
            elif category_counts[train_species_name]>=20:
                data_enhance(os.path.join(all_old_img_path, img_train), new_path,category='Two')
            elif category_counts[train_species_name]<6:
                data_enhance(os.path.join(all_old_img_path, img_train), new_path,category='Three')





