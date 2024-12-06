import os
import random

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1e-5

    def forward(self, predict, target):
        assert predict.size() == target.size(), "the size of predict and target must be equal."
        num = predict.size(0)
        # view和np.reshape作用相同：[N, C, H, W] -> [N, C*H*W]
        pre = torch.sigmoid(predict).view(num, -1)
        # 展平：[N, C, H, W] -> [N, C*H*W]
        tar = target.view(num, -1)
        intersection = (pre * tar).sum()  # 利用预测值与标签相乘当作交集
        union = (pre + tar).sum()
        dice = 1 - (2 * intersection + self.epsilon) / (union + self.epsilon)

        return dice


class SegmentationMetric(object):
    def __init__(self, num_class, num_batch):
        self.num_class = num_class
        # 初始化空混淆矩阵
        self.confusionMatrix = np.zeros((self.num_class, self.num_class))

    def reset(self):
        """
        初始化函数
        """
        self.confusionMatrix = np.zeros((self.num_class, self.num_class))

    def addBatch(self, predict, label):
        """
        同时添加整个batch的数据
        :param predict: 预测掩膜
        :param label: 真实掩膜
        :return:
        """
        assert predict.shape == label.shape
        for idx in range(label.shape[0]):
            # 累加混淆矩阵
            self.confusionMatrix += self.genConfusionMatrix(predict[idx], label[idx])
        return self.confusionMatrix

    def genConfusionMatrix(self, predict, label):  #
        """
        计算混淆矩阵
        :param predict: 预测掩膜
        :param label: 真实掩膜
        :return: 混淆矩阵: 十位是真实标记，个位是预测标记
        [[ 0,  1,  2,  3],
         [ 4,  5,  6,  7],
         [ 8,  9, 10, 11],
         [12, 13, 14, 15]]
        [[TN, FN],
         [FP, TP]]
        """
        # 过滤掉类别外掩膜，留下需要识别的掩膜
        label[label >= self.num_class] = 0
        # 巧妙利用self.num_class进制将对应信息存储于数字中：十位是真实标记，个位是预测标记
        label = self.num_class * label + predict
        # np.bincount计算非负整数数组中每个值的出现次数
        count = np.bincount(label.reshape(-1), minlength=self.num_class ** 2)
        # 输出数组是按从小到大每个整数升序排列
        confusionMatrix = count.reshape(self.num_class, self.num_class)
        return confusionMatrix

    def pixelAccuracy(self):
        """
        全局像素准确率：正确的像素占总像素的比例，混淆矩阵对角线列之和占全体之比，和分类任务中的准确率概念类似
        PA = (TP + TN) / (TP + TN + FP + TN)
        :return: 全局像素准确率
        """
        return np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()

    def classPixelAccuracy(self):
        """
        类别像素准确率：label中识别为某类像素中识别正确的比例
        cPA = (TP) / TP + FP
        :return: 返回的是一个列表值，该类别无像素时返回nan，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率
        """
        if self.confusionMatrix.sum(axis=0) == 0:
            return np.diag(self.confusionMatrix) / (self.confusionMatrix.sum(axis=0) + 1e-05)
        else:
            return np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=0)

    def meanPixelAccuracy(self):
        """
        平均像素准确率(MPA，Mean Pixel Accuracy)：是PA的一种简单提升，计算每个类内被正确分类像素数的比例，之后求所有类的平均。
        :return: 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89
        """
        return np.nanmean(self.classPixelAccuracy())

    def IntersectionOverUnion(self):
        """
        交并比（Intersection Over Union, IoU）:衡量的是预测的分割结果与真实标注（ground truth）之间的相似度
        IoU = TP / (TP + FP + FN)
        :return: 返回的是一个列表值，其值为各个类别的IoU，该类别无像素时返回nan
        """
        intersection = np.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)
        iou = intersection / union
        # 0类的IOU值弃掉
        return iou[1:]

    def meanIntersectionOverUnion(self):
        """
        平均交并比（Mean Intersection Over Union, mIoU）
        :return:
        """
        mIoU = np.nanmean(self.IntersectionOverUnion())  # 求各类别IoU的平均
        return mIoU

    def precision(self):
        """
        精准度 precision = TP / (TP+FP)
        :return: 返回列表值
        """
        return np.diag(self.confusionMatrix) / np.sum(self.confusionMatrix, axis=1)

    def recall(self):
        """
        召回率 recall = TP / (TP+FN)
        :return: 返回列表值
        """
        return np.diag(self.confusionMatrix) / np.sum(self.confusionMatrix, axis=0)

    def Frequency_Weighted_Intersection_over_Union(self):
        """
        FWIoU，频权交并比:为MIoU的一种提升，这种方法根据每个类出现的频率为其设置权重。
        FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        """
        freq = np.sum(self.confusionMatrix, axis=1) / np.sum(self.confusionMatrix)
        iu = np.diag(self.confusionMatrix) / (
                np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) -
                np.diag(self.confusionMatrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU


def convert_pixels_rgb(mask):
    """
    掩膜矩阵可视化
    :param mask: np.array[H, W],默认背景为0，所有类别的mask依次使用1，2，3作为掩膜标志
    :return: np.array[H, W, C],自动分色的uint8格式图片
    """
    classes = np.max(mask) + 1
    hsv_palette = np.array([[np.arange(classes) / classes * 180,
                             np.ones(classes) * 255, np.ones(classes) * 255]]).transpose((0, 2, 1))
    # 把背景设置为纯黑色
    hsv_palette[0, 0, :] = np.array([0, 0, 0])
    bgr_palette = cv2.cvtColor(np.uint8(hsv_palette), cv2.COLOR_HSV2BGR).squeeze(axis=0)
    mask_image = np.zeros(shape=(mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for idx in range(classes):
        mask_image[mask == idx] = bgr_palette[idx]

    return mask_image
