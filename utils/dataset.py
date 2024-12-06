# -*- coding: utf-8 -*-

import torch.utils.data as data
import os

import torch
import numpy as np
import cv2


class MyDataset(data.Dataset):
    def __init__(self, root, resize_h, resize_w, classes):
        """
        :param root: 数据集根目录，下属images和masks两个文件夹
        :param resize_h: 模型接受输入图像高
        :param resize_w: 模型接受输入图像宽
        """
        imgs = []
        img_list = os.listdir(f"{root}/images")
        img_list = sorted(img_list)
        label_list = os.listdir(f"{root}/masks")
        label_list = sorted(label_list)

        for i in range(len(img_list)):
            img = img_list[i]
            mask = label_list[i]
            imgs.append([f"{root}/images/{img}", f"{root}/masks/{mask}"])

        # N*2的列表，包含图像+掩膜路径的子列表
        self.imgs = imgs
        self.resize_h = resize_h
        self.resize_w = resize_w
        self.classes = classes

    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index]

        img = cv2.imread(img_path)
        # cv2读取多通道图片的方式比较特殊, 形状为[H, W, C]，需要转化为[C, H, W]
        img = cv2.resize(img, (self.resize_w, self.resize_h))
        img = np.transpose(img, axes=(2, 0, 1))
        # 转化为torch浮点值张量
        img = torch.from_numpy(img).type(torch.FloatTensor)

        # 读取npy格式的mask，并对选定标签进行二值化输出
        # 读取后, 形状为[H, W]
        mask = np.load(mask_path).astype(np.uint8)
        # cv2.resize不接受np.int32输入
        mask = cv2.resize(mask, (self.resize_w, self.resize_h), interpolation=cv2.INTER_NEAREST)
        # 转化为torch浮点值张量
        mask = torch.from_numpy(mask).type(torch.LongTensor)

        return img, mask

    def __len__(self):
        return len(self.imgs)


def one_hot_encoder(input, num_classes):
    """
    独热编码，训练时使用
    :param input: torch.tensor, 形状为[N, H, W]
    :param num_classes: 类别数num_classes
    :return: torch.tensor, 形状为[N, num_classes, H, W]
    """
    # 在位置1插入类别数
    shape = np.insert(arr=np.array(input.shape), obj=1, values=num_classes)
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(dim=1, index=torch.unsqueeze(input.cpu(), dim=1), value=1)


    return result
