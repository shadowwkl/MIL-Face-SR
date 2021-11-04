"""
-------------------------------------------------
   File Name:    __init__.py.py
   Author:       Zhonghao Huang
   Date:         2019/10/22
   Description:
-------------------------------------------------
"""

from data.datasets import FlatDirectoryImageDataset, FoldersDistributedDataset
from data.transforms import get_transform

import scipy.io as sio
import pdb
import numpy as np
import torch
# from torchvision.utils import save_image

def get_image_batches_test_random(data, all_labels, useful_labels, it, th):


    # r = np.random.RandomState(it)
    # select_view1_anr = np.random.permutation(2)[0]
    # if train is not True:
    #     it = it + 2500
    # pdb.set_trace()
    c_label = useful_labels[it%len(useful_labels)]
    c_label2 = useful_labels[np.random.permutation(len(useful_labels))[0]]
    # print(c_label-c_label2)
    image_idx = np.where(all_labels == c_label)[0]
    image_idx2 = np.where(all_labels == c_label2)[0]

    if len(image_idx) <= th:
        selected_idx = image_idx
    else:
        selected_idx = image_idx[np.random.permutation(len(image_idx))[0:th]]

    selected_idx2 = image_idx2[np.random.permutation(len(image_idx2))[0:th]]    

    image = torch.zeros([len(selected_idx), 3, 128, 128])
    image2 = torch.zeros([len(selected_idx), 3, 128, 128])

    # image = torch.zeros([10, 3, 128, 128])

    # pdb.set_trace()

    for i in range(len(selected_idx)):
        if i == 0:
            image[i] = data.dataset[selected_idx[i]]
        else:
            image[i] = data.dataset[selected_idx2[i]]

    for i in range(len(selected_idx)):
            image2[i] = data.dataset[selected_idx[i]]
    # for i in range(10):
    #     image[i] = data.dataset[i]
    # from torchvision.utils import save_image   
    # save_image(image,'ttt.jpg')

    # pdb.set_trace()

    return image, image2

def make_dataset(cfg):
    if cfg.folder:
        Dataset = FlatDirectoryImageDataset
    else:
        Dataset = FlatDirectoryImageDataset

    _dataset = Dataset(data_dir=cfg.img_dir, transform=get_transform(new_size=(cfg.resolution, cfg.resolution)))

    return _dataset


def get_data_loader(dataset, batch_size, num_workers):
    """
    generate the data_loader from the given dataset
    :param dataset: dataset for training (Should be a PyTorch dataset)
                    Make sure every item is an Image
    :param batch_size: batch size of the data
    :param num_workers: num of parallel readers
    :return: dl => data_loader for the dataset
    """
    from torch.utils.data import DataLoader

    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
        # pin_memory=True
    )

    return dl


def get_image_batches_test(data, all_labels, useful_labels, it, th):


    # r = np.random.RandomState(it)
    # select_view1_anr = np.random.permutation(2)[0]
    # if train is not True:
    #     it = it + 2500
    # pdb.set_trace()
    c_label = useful_labels[it%len(useful_labels)]
    # c_label = 1625
    image_idx = np.where(all_labels == c_label)[0]
    # np.random.seed(4)

    if len(image_idx) <= th:
        selected_idx = image_idx
    else:
        selected_idx = image_idx[np.random.permutation(len(image_idx))[0:th]]

    image = torch.zeros([len(selected_idx), 3, 128, 128])
    # selected_idx[-1] = 167625
    # selected_idx[0] = 175867
    # print(selected_idx)
    # pdb.set_trace()
    # image = torch.zeros([10, 3, 128, 128])


    # pdb.set_trace()

    for i in range(len(selected_idx)):
        image[i] = data.dataset[selected_idx[i]]
    # for i in range(10):
    #     image[i] = data.dataset[i]
    # from torchvision.utils import save_image   
    # save_image(image,'ttt.jpg')

    # pdb.set_trace()

    return image


def get_image_batches_test_mask(data, maskdata, all_labels, useful_labels, it, th):


    # r = np.random.RandomState(it)
    # select_view1_anr = np.random.permutation(2)[0]
    # if train is not True:
    #     it = it + 2500
    # pdb.set_trace()
    c_label = useful_labels[it%len(useful_labels)]
    image_idx = np.where(all_labels == c_label)[0]

    if len(image_idx) <= th:
        selected_idx = image_idx
    else:
        selected_idx = image_idx[np.random.permutation(len(image_idx))[0:th]]

    image = torch.zeros([len(selected_idx), 3, 128, 128])
    mask = torch.zeros([len(selected_idx), 1, 128, 128])
    # image = torch.zeros([10, 3, 128, 128])

    # pdb.set_trace()

    for i in range(len(selected_idx)):
        image[i] = data.dataset[selected_idx[i]]
        mask[i] =  maskdata.dataset[selected_idx[i]][0,:,:]
    # for i in range(10):
    #     image[i] = data.dataset[i]
    # from torchvision.utils import save_image   
    # save_image(image,'ttt.jpg')

    # pdb.set_trace()

    return image, mask.cuda()

def get_image_batches(data, all_labels, useful_labels, it, th, batchSize, r_idx):


    # r = np.random.RandomState(it)
    # select_view1_anr = np.random.permutation(2)[0]
    # if train is not True:
    #     it = it + 2500
    # pdb.set_trace()
    image = torch.zeros([batchSize*th, 3, 128, 128])

    for j in range(batchSize):
        # pdb.set_trace()
        # print((it*batchSize)+j)
        c_label = useful_labels[r_idx[((it*batchSize)+j)]%len(useful_labels)]
        image_idx = np.where(all_labels == c_label)[0]

        if len(image_idx) <= th:
            selected_idx = image_idx
        else:
            selected_idx = image_idx[np.random.permutation(len(image_idx))[0:th]]

    # image = torch.zeros([10, 3, 128, 128])

        # pdb.set_trace()

        for i in range(len(selected_idx)):
            # print(len(selected_idx)*j+i)
            image[len(selected_idx)*j+i] = data.dataset[selected_idx[i]]
    # for i in range(10):
    #     image[i] = data.dataset[i]
    # from torchvision.utils import save_image   
    # save_image(image,'ttt.jpg')

    # pdb.set_trace()

    # print(image.shape)
    # pdb.set_trace()
    return image
  




    # person_sel = np.random.permutation(train_person_num)[0:2]

    # person_1=np.where(label_train_cam1==person_sel[0])[0];
    # person_2=np.where(label_train_cam1==person_sel[1])[0]; 
    # # pdb.set_trace()
    # # person_1 = person_1[np.random.permutation(len(person_1))[0:np.min([8, len(person_1)])]]
    # # person_1 = person_1[np.random.permutation(len(person_1))[0:np.min([10, len(person_1)])]]
    # person_1 = person_1[np.random.permutation(len(person_1))[0:5]]



    # pos_image = torch.zeros([len(person_1), 3, 224, 224])
    # neg_image = torch.zeros([len(person_1), 3, 224, 224])


    # for i in range(len(person_1)):
    #     pos_image[i] = F.interpolate(train_data_cam1.dataset[person_1[i]].unsqueeze(0), size=(224, 224), mode='bilinear')
    #     neg_image[i] = F.interpolate(train_data_cam1.dataset[person_1[i]].unsqueeze(0), size=(224, 224), mode='bilinear')

    # # neg_image = pos_image
    # # pdb.set_trace()
    # randomplace = np.random.randint(len(person_1))
    # neg_image[randomplace,:,:,:] = F.interpolate(train_data_cam1.dataset[person_2[np.random.randint(len(person_2))]].unsqueeze(0), size=(224, 224), mode='bilinear')


    # # pdb.set_trace()
   
    # return pos_image
