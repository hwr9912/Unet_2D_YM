# -*- coding:utf-8 -*-
# 此模块为所有的指标都输出


import argparse
import numpy as np
import os
import time
from tqdm import tqdm
import datetime  # 获取指定日期和时间

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils import data

from utils.dataset import MyDataset, one_hot_encoder
from utils.utils import DiceLoss, SegmentationMetric
from utils.model_builder import build_model


def train_val_cycle(args):
    # 将模型加载至GPU
    model = build_model(args.model, args.classes).to(args.device)
    # 加载训练集及测试集
    train_loader, test_loader = data_loader(args)
    # 保存模型在验证集上的表现
    results_file = f"{args.model}_{args.dataset}_results.txt"
    # 模型参数保存路径，不存在则创建
    weights_save_path = f"{args.save_dir}/{args.model}_{args.dataset}"
    if not os.path.exists(weights_save_path):
        os.makedirs(weights_save_path)

    # 加载模型继续训练
    # 如果选择加载模型继续训练，且文件存在
    if args.resume and os.path.isfile(args.resume):
        model.load_state_dict(torch.load(args.resume))
        print(f"Loaded checkpoint '{args.resume}'! ")
        start_epoch = int(os.path.basename(args.resume).split(".")[0].split("_")[1])
    elif args.resume:
        print(f"No checkpoint found at '{args.resume}'!")
        start_epoch = 0
    else:
        print(f"Train from the ground!")
        start_epoch = 0
    # 模型日志记录
    writer = SummaryWriter(args.directory)

    # 损失函数
    criterion = DiceLoss()
    # 优化函数
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # 学习率调度器：在指定milestone，以现有的学习率乘上给定衰减因子得到新学习率
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones,
                                                     gamma=args.lr_scheduler_gamma)

    for epoch in range(start_epoch, args.epochs):
        # 模型训练 =============================================================================================
        #   训练时间记录
        train_start_time = time.time()
        average_epoch_loss_train = train(args, model, train_loader, optimizer, criterion)
        train_end_time = time.time()
        #   屏幕输出每轮训练损失及耗时
        print(f"Epoch:{epoch + 1:3d}\t"
              f"train_loss:{average_epoch_loss_train:4f}\t"
              f"Time:{(train_end_time - train_start_time) / 60:2f}min")
        writer.add_scalar("Train Loss", average_epoch_loss_train, epoch + 1)
        scheduler.step()
        if epoch != 0 and epoch % 100 == 0:  # model.state_dict()只保存训练好的权重，dir绝对路径加文件名
            torch.save(model.state_dict(),f"{weights_save_path}/{args.model}_{epoch}.pth")
        elif epoch == args.epochs - 1:  # model.state_dict()只保存训练好的权重，dir绝对路径加文件名
            torch.save(model.state_dict(), f"{weights_save_path}/{args.model}_finished.pth")

        # 模型评估 =============================================================================================
        if epoch % 1 == 0 or epoch == (args.epochs - 1):
            val_start_time = time.time()
            # 计算模型评估指标
            eval_loss, pa_, mPA_, IOU_, mIOU_, Pr_, Recall_, F1_ = val(args, model, test_loader, criterion)
            val_end_time = time.time()

            print("Epoch:{:3d}"
                  "\tEval_loss:{:4f}"  # \t代表一个tab
                  "\tpa:{:.4f}"
                  "\tmPA:{:.4f}"
                  "\tIOU:{}"
                  "\nmIOU:{:.4f}"
                  "\tPrecision:{:.4f}"
                  "\tRecall:{:.4f}"
                  "\tF1 Score:{:.4f}"
                  "\tTime:{:2f}min".format(epoch + 1, eval_loss, pa_, mPA_, IOU_, mIOU_, Pr_, Recall_, F1_,
                                           (val_end_time - val_start_time) / 60))

            writer.add_scalar("Eval Loss", eval_loss, epoch + 1)  # 名称，值，横坐标
            writer.add_scalar("mPA", mPA_, epoch + 1)
            writer.add_scalar("mIOU", mIOU_, epoch + 1)
            writer.add_scalar("Precision", Pr_, epoch + 1)
            writer.add_scalar("Recall", Recall_, epoch + 1)
            writer.add_scalar("F1 Score", F1_, epoch + 1)

            with open(os.path.join(weights_save_path, results_file), "a") as f:
                # 记录每个epoch对应的train_loss、lr以及验证集各指标，mean_loss训练过程平均损失
                # python数字格式化方法，:.4f保留小数点后四位
                train_info = (f"[epoch: {epoch + 1}]\n"
                              f"loss: {average_epoch_loss_train}\n"
                              f"time: {(train_end_time - train_start_time) / 60}\n")
                val_info = (f"epoch:{epoch + 1:3d}"
                            f"\tEval_loss:{eval_loss:4f}"
                            f"\tpa:{pa_:.4f}"
                            f"\tmPA:{mPA_:.4f}"
                            f"\tIOU:{IOU_}"
                            f"\nmIOU:{mIOU_:.4f}"
                            f"\tPrecision:{Pr_:.4f}"
                            f"\tRecall:{Recall_:.4f}"
                            f"\tF1 Score:{F1_:.4f}"
                            f"\tTime:{(val_end_time - val_start_time) / 60:2f}min")

                f.write(train_info + val_info + "\n\n")

    writer.close()
    print(f"Model saved at {weights_save_path}.")


def train(args, model, train_loader, optimizer, criterion):
    # 训练模式
    model.train()
    # 初始化参数
    epoch_loss = []
    for x, y in tqdm(train_loader):
        # 加载图像数据：[N, C, H, W]
        inputs = x.to(args.device)
        # 对y进行独热编码：[N, H, W] -> [N, num_classes, H, W]
        labels = one_hot_encoder(y, args.classes).to(args.device)
        # 梯度清零
        optimizer.zero_grad()
        # 正向传播
        outputs = model(inputs)
        # 反向传播求梯度
        loss = criterion(outputs, labels)
        loss.backward()
        # 权重更新
        optimizer.step()
        epoch_loss.append(loss.item())

    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)

    return average_epoch_loss_train


def val(args, model, test_loader, criterion):
    # 评估模式
    model.eval()
    # 初始化参数
    eval_loss = 0.0
    PA, cPA, mPA, IOU, mIOU, Pr, Recall, F1 = 0, 0, 0, 0, 0, 0, 0, 0
    test_loader_size = len(test_loader)
    # 预测效能评估
    metric = SegmentationMetric(num_class=args.classes, num_batch=test_loader_size)
    with torch.no_grad():
        for x, y in tqdm(test_loader):
            # 加载图像数据：[N, C, H, W]
            inputs = x.to(args.device)
            # 对y进行独热编码：[N, H, W] -> [N, num_classes, H, W]
            labels = one_hot_encoder(y, args.classes).to(args.device)
            # 正向传播
            outputs = model(inputs)
            # 计算损失函数
            loss = criterion(outputs, labels)
            # 平均损失
            eval_loss += loss.item()
            # 转化为numpy:[N, num_classes, H, W]
            predict = outputs.data.cpu().numpy()
            # 独热编码转类别代码:[N, num_classes, H, W] -> [N, H, W]
            predict = np.argmax(predict, axis=1)
            # 转化为numpy:[N, H, W]
            target = y.numpy()

            # 更新混淆矩阵
            cm = metric.addBatch(predict, target)
            # 计算参数
            pa = metric.pixelAccuracy()
            # cpa = metric.classPixelAccuracy()
            mpa = metric.meanPixelAccuracy()
            IoU = metric.IntersectionOverUnion()
            mIoU = metric.meanIntersectionOverUnion()
            precision = metric.precision()
            recall = metric.recall()
            f1 = (2 * precision * recall) / (precision + recall)

            pr_list = []
            rcl_list = []
            f1_list = []
            for i in range(len(precision)):
                if not np.isnan(precision[i]):
                    pr_list.append(precision[i])
                if not np.isnan(recall[i]):
                    rcl_list.append(recall[i])
                if not np.isnan(f1[i]):
                    f1_list.append(f1[i])
            pa += pa
            mPA += mpa
            IOU += IoU
            mIOU += mIoU
            Pr += np.mean(pr_list)
            Recall += np.mean(rcl_list)
            F1 += np.mean(f1_list)

    pa_ = pa / test_loader_size
    mPA_ = mPA / test_loader_size
    IOU_ = IOU / test_loader_size
    mIOU_ = mIOU / test_loader_size
    Pr_ = Pr / test_loader_size
    Recall_ = Recall / test_loader_size
    F1_ = F1 / test_loader_size

    return eval_loss, pa_, mPA_, IOU_, mIOU_, Pr_, Recall_, F1_


def data_loader(args):
    train_dataset = MyDataset(root=f"data/{args.dataset}",
                              resize_h=args.size[0],
                              resize_w=args.size[1],
                              classes=args.classes)
    n_val = int(len(train_dataset) * args.train_test_ratio)
    n_train = len(train_dataset) - n_val
    train, val = data.random_split(train_dataset, [n_train, n_val])
    # num_workers: 同时使用的进程数
    # drop_last: 是否丢弃最后一批数据
    # pin_memory: 固定内存
    train_loader = data.DataLoader(dataset=train, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers, drop_last=True, pin_memory=True)
    test_loader = data.DataLoader(dataset=val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                  drop_last=False, pin_memory=True)

    return train_loader, test_loader


def parse_args():
    parser = argparse.ArgumentParser(description="Semantic Segmentation")
    parser.add_argument("--model", type=str, default='unet')
    parser.add_argument("--dataset", type=str, default="dataset_0")
    parser.add_argument("--train_test_ratio", type=int, default=0.1)
    parser.add_argument("--classes", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--size", type=tuple, default=(256, 256), help="(H, W)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_lr", type=float, default=0.012)
    parser.add_argument("--min_lr", type=float, default=1e-4)
    parser.add_argument("--milestones", type=list, default=[100, 200, 500])
    parser.add_argument("--lr_scheduler_gamma", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument("--momentum", type=float, default=0.99)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str, default="weights")
    parser.add_argument("--resume", type=str, default=r"", help="Load checkpoint for continuing training.")
    parser.add_argument('--experiment-start-time', type=str,
                        default=datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S'))
    args = parser.parse_args()

    directory = f"runs/{args.model}_{args.dataset}_{args.experiment_start_time}"

    args.directory = directory
    return args


if __name__ == '__main__':
    args = parse_args()
    print("args: ", args, "\n")
    train_val_cycle(args)
