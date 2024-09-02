import numpy as np
import os
import torch
from PIL import Image
import torchvision.transforms as transforms

def class_count_guided_data_loader(path,bins,resize=None,is_category=False):
    assert  len(bins)>0 and type(bins)==list
    X_samples = []
    y_samples = []
    category = []
    class_weights = []
    for i in range(len(bins)):
        X_samples.append([])
        y_samples.append([])
        category.append({})
        class_weights.append([])
    if os.path.isdir(path):
        for label,filename in enumerate(os.listdir(path)):
            file_path = os.path.join(path,filename)
            if os.path.isdir(file_path):
                data_count = len(os.listdir(file_path))
                for i in range(len(bins)):
                    if data_count >= bins[i] and data_count < (bins[i+1] if i+1 <=len(bins) - 1 else float('inf')):
                        if is_category:
                            category[i].update({label: data_count})
                        for image in os.listdir(file_path):
                            image = Image.open(os.path.join(file_path,image)).convert('RGB')
                            if resize is not None:
                                image = image.resize(resize)
                            X_samples[i].append(transforms.ToTensor()(image))
                            y_samples[i].append(label)
    for i in range(len(X_samples)):
        X_samples[i] = torch.stack(X_samples[i]).float()
        y_samples[i] = torch.tensor(y_samples[i]).long()
        class_weights[i] = torch.ones(y_samples[i].shape).float()
    return X_samples,y_samples,class_weights,category
