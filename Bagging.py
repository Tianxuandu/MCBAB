import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image
from LeNet import LeNet
from AlexNet import AlexNet
import torchvision
from torchvision import transforms
from tqdm import tqdm
import wandb
import Bagging_Adaboosting
import torchvision.models as models

#wandb.login(key='ddd005d13a9704b2f25fd1c6ace472b6ca714fc2')

#数据加载器
def data_loader(data_dir,resize=None,is_category = False):
    #创建features与labels列表
    X = []
    y = []
    class_weights = []
    category = {}
    if os.path.exists(data_dir):
        for label,filename in enumerate(os .listdir(data_dir)):
            file_path = os.path.join(data_dir,filename)
            if os.path.isdir(file_path):
                if is_category:
                    category.update({label:sum(os.path.isfile(os.path.join(file_path,entery)) for entery in os.listdir(file_path))})
                for image in os.listdir(file_path):
                    image_path = os.path.join(file_path,image)
                    image = Image.open(image_path).convert('RGB')
                    if resize is not None:
                        image = image.resize(resize)
                    X.append(transforms.ToTensor()(image))
                    y.append(label)
        X = torch.stack(X).float()
        y_one_hot = torch.tensor(y).long()#.float()
        class_weights = torch.ones(y_one_hot.shape)
        #y_one_hot = F.one_hot(torch.tensor(np.array(y),dtype=torch.long),num_classes=len(category)).float()
    else:
        y_one_hot = F.one_hot(torch.tensor(np.array(y),dtype=torch.long),num_classes=1).float()
        class_weights = torch.ones(y_one_hot.shape)
        print('data_loader error!!!!')
    return X, y_one_hot,class_weights,category

#bagging自助抽样
def Bootstrap_samples(X_samples,y_samples,replace=True):
    #确保X和y是来自同一个数据集
    assert X_samples.shape[0] == y_samples.shape[0]
    samples_id = np.random.choice(len(X_samples),len(X_samples),replace=replace)
    X_new_samples = X_samples[samples_id]
    y_new_samples = y_samples[samples_id]
    return X_new_samples,y_new_samples

#自定义采样
'''每次从每个类别中随机采样n张照片组成新的数据集'''
def random_balance_samples(X_samples,y_samples,n_per_datas,category=None):
    assert X_samples.shape[0] == y_samples.shape[0]
    assert all(n_per_datas < x for x in list(category.values()))
    assert type(category) == dict
    X_new_samples = []
    y_new_samples = []
    start_n = 0
    for j in list(category.values()):
        sample_id = np.random.choice(j,n_per_datas)
        sample_id = [sample_index+start_n for sample_index in sample_id]
        for sample_idx in sample_id:
            X_new_samples.append(X_samples[sample_idx].tolist())
            y_new_samples.append(y_samples[sample_idx].tolist())
        start_n += j

    return torch.tensor(X_new_samples),torch.tensor(y_new_samples)

#批量生成器
def batch_data_loader(X_samples,y_samples,batch_size,class_weights,is_shuffle=False):
    num_samples = len(X_samples)
    indices = list(range(num_samples))
    if is_shuffle:
        np.random.shuffle(indices)
    else:
        pass
    for i in range(0,num_samples,batch_size):
        batch_indices = torch.tensor(indices[i:min(i+batch_size,num_samples)])
        yield X_samples[batch_indices],y_samples[batch_indices],class_weights[batch_indices]

#模型生成器
def models_list(n_models,num_classes):
    #model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    #model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = LeNet(3,5)  #本机测试用的
    models = []
    models_request = []   #构建列表，存储resize，num_classes，in_channels数值
    model2 = AlexNet(5)
    models_request.append([(224,224),num_classes,3])
    models_request.append([(224, 224), num_classes, 3])
    models.append(model2)
    models.append(model2)
    for _ in range(0,n_models-2):
        models.append(model)
        models_request.append([(28,28),num_classes,3])
    return models , models_request

#评估器
def final_accuacy(final_test_list,test_labels,add_list_dict):
    for key,values in add_list_dict.items():
        i,j = map(int,key.split(','))
        del final_test_list[j][-values:]
        break

    final_final_test_list = []

    for i in range(len(final_test_list)):
        for j in range(len(final_test_list[i])):
            final_final_test_list.append(final_test_list[i][j])
    print(len(test_labels),len(final_final_test_list))
    assert type(test_labels) == type(final_final_test_list) == list
    assert len(test_labels) == len(final_final_test_list)

    final_final_test_list = torch.tensor(final_final_test_list).float()
    test_labels = torch.tensor(test_labels).float()
    _,predicted = torch.max(final_final_test_list,dim=1)
    #_,truth = torch.max(test_labels,dim=1)
    correct = (predicted==test_labels).sum().item()
    total = len(test_labels)
    final_accuacy = correct/total
    return final_accuacy

def is_iterable(variable):
    try:
        iter(variable)
        return True
    except TypeError:
        return False

#bagging均权投票
def result_optimiszer_selective_bagging_Ada(workers_test_list,num_classes):
    assert type(workers_test_list) == list and len(workers_test_list) > 0
    probs = []
    add_list_dict = {}
    for i in range(len(workers_test_list)):
        max_length = max(len(sublist) for sublist in workers_test_list[i])
        for j in range(len(workers_test_list[i])):
            if len(workers_test_list[i][j]) < max_length:
                add_list_dict[f'{i},{j}'] = max_length - len(workers_test_list[i][j])
                workers_test_list[i][j] += [[0] * num_classes] * (max_length - len(workers_test_list[i][j]))

    for i in range(len(workers_test_list)):
        probs1 = []
        for j in range(len(workers_test_list[0])):
            probs2 = []
            for k in range((len(workers_test_list[0][0]))):
                max_category = np.argmax(np.array(workers_test_list[i][j][k]))
                probs_small = np.zeros(len(workers_test_list[i][j][k]),dtype=int)
                probs_small[max_category] = 1
                probs2.append(probs_small.tolist())
            probs1.append(probs2)
        probs.append(probs1)
    final_predict = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)),axis=0,
                                        arr=np.array(probs)).reshape(np.array(workers_test_list[0]).shape)

#bagging权值投票
#权值只有0和1，且纯标签后处理，权值并非模型权值
def result_optimiszer_selective(workers_test_list,num_classes,class_weights):
    assert type(workers_test_list) == list and len(workers_test_list) > 0
    probs = []
    add_list_dict = {}
    for i in range(len(workers_test_list)):
        max_length = max(len(sublist) for sublist in workers_test_list[i])
        for j in range(len(workers_test_list[i])):
            if len(workers_test_list[i][j]) < max_length:
                add_list_dict[f'{i},{j}'] = max_length - len(workers_test_list[i][j])
                for _ in range(max_length - len(workers_test_list[i][j])):
                    class_weights[i][j].append(0.0)
                workers_test_list[i][j] += [[0] * num_classes] * (max_length - len(workers_test_list[i][j]))


    for i in range(len(workers_test_list)):
        probs1 = []
        for j in range(len(workers_test_list[0])):
            probs2 = []
            for k in range((len(workers_test_list[0][0]))):
                max_category = np.argmax(np.array(workers_test_list[i][j][k]))
                probs_small = np.zeros(len(workers_test_list[i][j][k]),dtype=int)
                probs_small[max_category] = 1
                probs2.append(probs_small.tolist())
            probs1.append(probs2)
        probs.append(probs1)
    final_predict = []
    print(probs)
    for i in range(len(probs[0])):
        for j in range(len(probs[0][0])):
            probs_0d_list = []
            weights_0d_list = []
            for k in range(len(probs)):
                probs_0d_list.append(probs[k][i][j])
                weights_0d_list.append(class_weights[k][i][j])
            final_predict.append(weights_cal_result(weights=weights_0d_list,result_list=probs_0d_list))
    final_predict = torch.tensor(final_predict).reshape(torch.tensor(workers_test_list[0]).shape)
    return final_predict.tolist(),add_list_dict

#bagging权值输出函数
def weights_cal_result(weights,result_list):
    assert type(weights) == type(result_list) == list
    assert len(weights) == len(result_list)
    work_list = []
    for i in range(len(weights)):
        if weights[i] == 1:
            work_list.append(result_list[i])
        else:
            continue
    if work_list != []:
        return work_list
    else:
        return np.zeros(torch.tensor(work_list).shape).tolist()

def data_iter_product(n_iters,epochs,models,models_request,batch_size,samples_method='balance'):
    assert type(models) == list and  n_iters == len(models) == len(models_request)
    data_dir_train = 'C:/Users/Dumin/Desktop/dataset/train'
    data_dir_test = 'C:/Users/Dumin/Desktop/dataset/test'
    train_iter_list = []
    test_iter_list = []
    resize_list = []
    X_samples_dict = {}
    y_samples_dict = {}
    X_samples_test_dict = {}
    y_samples_test_dict = {}
    class_weights_train_dict = {}
    class_weights_test_dict = {}
    category_dict = {}
    for i in range(n_iters):
        resize_list.append(models_request[i][0])
    resize_list = set(resize_list)
    for resize in resize_list:
        X_samples_dict[f'{resize}'], y_samples_dict[f'{resize}'], class_weights_train_dict[f'{resize}'], category_dict[f'{resize}'] = data_loader(
            data_dir=data_dir_train,
            resize=resize,
            is_category=True)
        X_samples_test_dict[f'{resize}'], y_samples_test_dict[f'{resize}'], class_weights_test_dict[f'{resize}'], _ = data_loader(
            data_dir=data_dir_test,
            resize=resize,
            is_category=True)
    for model in models:
        if samples_method == 'balance':
            X_samples, y_samples = random_balance_samples(
                X_samples=X_samples_dict[f'{models_request[models.index(model)][0]}'],
                y_samples=y_samples_dict[f'{models_request[models.index(model)][0]}'],
                n_per_datas=10,
                category=category_dict[f'{models_request[models.index(model)][0]}']
            )
        elif samples_method == 'bootstrap':
            X_samples, y_samples = Bootstrap_samples(
                X_samples=X_samples_dict[f'{models_request[models.index(model)][0]}'],
                y_samples=y_samples_dict[f'{models_request[models.index(model)][0]}']
            )
        else:
            print('train data loader error!!!')
            break
        train_per_model_iter_list = []
        test_per_model_iter_list = []
        for epoch in range(epochs):
            train_loader = batch_data_loader(
                X_samples=X_samples,
                y_samples=y_samples,
                batch_size=batch_size,
                class_weights=class_weights_train_dict[f'{models_request[models.index(model)][0]}']
            )

            test_loader = batch_data_loader(
                X_samples=X_samples_test_dict[f'{models_request[models.index(model)][0]}'],
                y_samples=y_samples_test_dict[f'{models_request[models.index(model)][0]}'],
                batch_size=batch_size,
                class_weights=class_weights_test_dict[f'{models_request[models.index(model)][0]}']
            )
            train_per_model_iter_list.append(train_loader)
            test_per_model_iter_list.append(test_loader)
        train_iter_list.append(train_per_model_iter_list)
        test_iter_list.append(test_per_model_iter_list)
    return train_iter_list,test_iter_list

def train_model(n_base_models,n_samples,epochs,batch_size,lr,weight_decay,num_classes,cuda_id=None):
    assert n_base_models == n_samples
    models,models_request = models_list(n_base_models,num_classes=num_classes)
    test_labels_list = []
    test_labels_class_weights = []
    for model in tqdm(models,desc='models'):
        wandb.log({
            'model':model,
            'model_index':models.index(model),
        })
        model_index = models.index(model)
        if cuda_id is not None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = nn.DataParallel(model,device_ids=cuda_id).to(device)
        else:
            device = torch.device( 'cpu')
        test_one_model = []
        test_one_class_weights = []
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        train_loader,test_loader = data_iter_product(
            n_iters=n_base_models,
            epochs=epochs,
            models=models,
            models_request=models_request,
            batch_size=batch_size,
            samples_method='bootstrap'
        )
        for epoch in tqdm(range(epochs),desc='train epochs'):
            train_total = 0
            train_correct = 0
            train_iter = iter(train_loader[model_index][epoch])
            for X,y,class_weights_train_batch in tqdm(train_iter,desc='train batch'):
                model.train()
                optimizer.zero_grad()
                if cuda_id is not None:
                    X ,y,class_weights_train_batch= X.to(device),y.to(device),class_weights_train_batch.to(device)
                    y_hat = model(X)
                    train_loss = loss(y_hat,y)
                else:
                    y_hat = model(X)
                    train_loss = loss(y_hat,y)
                train_loss.backward()
                train_total += y.shape[0]
                _,predicted = torch.max(y_hat.data,1)
                for i in range(len(y_hat)):
                    if predicted[i]==y[i]:
                        class_weights_train_batch[i] = 1
                    else:
                        class_weights_train_batch[i] = 0
                #_,truth = torch.max(y.data,1)
                train_correct += (predicted==y).sum().item()
                optimizer.step()
            train_acc_global = 100.* train_correct / train_total

            wandb.log({
                'epoch':epoch+1,
                'train_loss': train_loss.item(),
                'train_acc': train_acc_global/100
                       })

            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                y_hat_list = []
                test_one_class_weights_temp = []
                test_accuacy_list = [0]
                test_iter = iter(test_loader[model_index][epoch])
                for X,y,class_weights_test_batch in tqdm(test_iter,desc='test batch'):
                    if cuda_id is not None:
                        X,y,class_weights_test_batch = X.to(device),y.to(device),class_weights_test_batch.to(device)
                    else:
                        X,y,class_weights_test_batch=X,y,class_weights_test_batch
                    y_hat = model(X)
                    test_loss = loss(y_hat,y)
                    y_hat_list.append(y_hat.tolist())
                    _,predicted = torch.max(y_hat.data,1)
                    for i in range(len(y_hat)):
                        if predicted[i] == y[i]:
                            class_weights_test_batch[i] = 1
                        else:
                            class_weights_test_batch[i] = 0
                    test_one_class_weights_temp.append(class_weights_test_batch.tolist())
                    #_, truth = torch.max(y.data, 1)
                    total += y.shape[0]
                    correct += (predicted==y).sum().item()
                test_accuacy = correct / total
                test_accuacy_list.append(test_accuacy)
                if all(test_accuacy >= x for x in test_accuacy_list):
                    test_one_model = y_hat_list
                    test_one_class_weights = test_one_class_weights_temp
                else:
                    pass
                wandb.log({
                    'test_loss': test_loss.item(),
                    'test_acc': correct / total
                })
            print(f'epochs:{epoch+1}/{epochs},test_loss:{test_loss:.4f},test_acc:{correct/total*100:.4f}%')
    test_labels_list.append(test_one_model)
    test_labels_class_weights.append(test_one_class_weights)
    return result_optimiszer_selective(test_labels_list,num_classes=len(test_labels_list[0][0][0]),class_weights=test_labels_class_weights)