# -*- coding:utf-8 -*-

import torch
import numpy as np
import os
import cv2
from utils.utils import convert_pixels_rgb
import argparse
from utils.model_builder import build_model


def inference(args):
    # 构建模型并加载参数
    model = build_model(args.model, args.classes).to(args.device)
    model.load_state_dict(torch.load(args.model_params))
    # 创建预测掩膜输出路径
    output_dir = f"{args.output_dir}/{args.model}"
    os.makedirs(output_dir, exist_ok=True)
    # 设置为评估模式
    model.eval()
    for f in os.listdir(args.dataset):
        img_path = f"{args.dataset}/{f}"
        try:
            input = cv2.imread(img_path)
            # cv2读取多通道图片的方式比较特殊, 形状为[H, W, C]，需要转化为[C, H, W]
            input = cv2.resize(input, (args.size[1], args.size[0]))
            input = np.transpose(input, axes=(2, 0, 1))
            # 转化为torch浮点值张量
            input = torch.from_numpy(input).type(torch.FloatTensor).to(args.device)
            # 把[C, H, W]转化为可输入的[1, C, H, W]
            input = torch.unsqueeze(input, dim=0)

            # 输出为[1, num_classes, H, W]
            output = model(input)
            predict = output.data.cpu().numpy()
            # 独热编码转分类掩膜: [1, num_classes, H, W] -> [1, H, W]
            predict = np.argmax(predict, axis=1)
            # [1, H, W] -> [H, W]
            predict = predict[0, :, :]
            # 掩膜转图片：[H, W] -> [H, W, C]
            converted_image = convert_pixels_rgb(predict)
            # 输出图片
            cv2.imwrite(f"{output_dir}/{f}", converted_image)

        except Exception as Error:
            print(Error)


def parse_args():
    parser = argparse.ArgumentParser(description="Semantic Segmentation")
    parser.add_argument("--model", type=str, default="unet")
    parser.add_argument("--model_params", type=str, default="weights/unet_dataset_0/unet_finished.pth")
    parser.add_argument("--dataset", type=str, default="data/val/image")
    parser.add_argument("--output_dir", type=str, default=r"data/val/mask")
    parser.add_argument("--classes", type=int, default=2)
    parser.add_argument("--size", type=tuple, default=(256, 256), help="(H, W)")
    parser.add_argument("--device", type=str, default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print("args: ", args)
    inference(args)
