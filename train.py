# -*- coding: utf-8 -*-

import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
import datetime
import numpy as np
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split
import joblib
from skimage.io import imread

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms
from dataset.dataset import Dataset
import matplotlib.pyplot as plt
from utilities.metrics import dice_coef, batch_iou, mean_iou, iou_score
import utilities.losses as losses
from utilities.utils import str2bool, count_params
import pandas as pd
from net import Unet,unet,qau_net
from  utilities.lr_policy import  PolyLR
# from  net.small_ecanet import PraNet121
#换模型需要修改的地方
arch_names = list(Unet.__dict__.keys())
loss_names = list(losses.__dict__.keys())
loss_names.append('BCEWithLogitsLoss')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    
    #换模型需要修改的地方
    parser.add_argument('--arch', '-a', metavar='ARCH', default='Unet',
                        choices=arch_names,
                        help='model architecture: ' +
                            ' | '.join(arch_names) +
                            ' (default: NestedUNet)')
    #换数据集需要修改的地方
    parser.add_argument('--dataset', default="LiTS",
                        help='dataset name')
    parser.add_argument('--input-channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--image-ext', default='npy',
                        help='image file extension')
    parser.add_argument('--mask-ext', default='npy',
                        help='mask file extension')
    parser.add_argument('--aug', default=False, type=str2bool)
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=loss_names,
                        help='loss: ' +
                            ' | '.join(loss_names) +
                            ' (default: BCEDiceLoss)')
    #换模型需要修改的地方
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--early-stop', default=50, type=int,
                        metavar='N', help='early stopping (default: 30)')
    
    #换模型需要修改的地方
    parser.add_argument('-b', '--batch-size', default=4, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                            ' | '.join(['Adam', 'SGD']) +
                            ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')
    parser.add_argument('--deepsupervision', default=False, type=str2bool,
                        help='nesterov')

    args = parser.parse_args()

    return args

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(args, train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    ious = AverageMeter()
    dices_1s = AverageMeter()
    dices_2s = AverageMeter()
    model.train()

    for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        input = input.cuda()
        target = target.cuda()
        # compute output
        if args.deepsupervision:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            iou = iou_score(outputs[-1], target)
        else:
            output = model(input)
            loss = criterion(output, target)
            iou = iou_score(output, target)
            dice_1 = dice_coef(output, target)[0]
            dice_2 = dice_coef(output, target)[1]

        losses.update(loss.item(), input.size(0))
        ious.update(iou, input.size(0))
        dices_1s.update(torch.tensor(dice_1), input.size(0))
        dices_2s.update(torch.tensor(dice_2), input.size(0))

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
        ('dice_1', dices_1s.avg),
        ('dice_2', dices_2s.avg)
    ])

    return log


def validate(args, val_loader, model, criterion):
    losses = AverageMeter()
    ious = AverageMeter()
    dices_1s = AverageMeter()
    dices_2s = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target,filename) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            # compute output
            if args.deepsupervision:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou = iou_score(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou = iou_score(output, target)
                dice_1 = dice_coef(output, target)[0]
                dice_2 = dice_coef(output, target)[1]
                output = torch.sigmoid(output)
                pred_liver = output.detach().cpu().numpy()[0][0]
                pred_tumor = output.detach().cpu().numpy()[0][1]
                pred_liver[pred_liver > 0.5] = 255
                pred_liver[pred_liver <= 0.5] = 0
                pred_tumor[pred_tumor > 0.5] = 255
                pred_tumor[pred_tumor <= 0.5] = 0
                pred_liver[pred_tumor>1] = 133
                im = Image.fromarray(np.uint8(pred_liver))
                im.convert('L').save("/home/luosy/TTNet/pred/tmp/"+filename[0]+'.jpg')

            losses.update(loss.item(), input.size(0))
            ious.update(iou, input.size(0))
            dices_1s.update(torch.tensor(dice_1), input.size(0))
            dices_2s.update(torch.tensor(dice_2), input.size(0))

    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
        ('dice_1', dices_1s.avg),
        ('dice_2', dices_2s.avg)
    ])

    return log


def main():
    args = parse_args()
    #args.dataset = "datasets"

    if args.name is None:
        if args.deepsupervision:
            args.name = '%s_%s_lym' %(args.dataset, args.arch)
        else:
            args.name = '%s_%s_lym' %(args.dataset, args.arch)
    timestamp  = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if not os.path.exists('models/{}/{}'.format(args.name,timestamp)):
        os.makedirs('models/{}/{}'.format(args.name,timestamp))

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    with open('models/{}/{}/args.txt'.format(args.name,timestamp), 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/{}/{}/args.pkl'.format(args.name,timestamp))

    # define loss function (criterion)
    if args.loss == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.BCEDiceLoss().cuda()
    cudnn.benchmark = True

    # Data loading code
    train_img_paths = glob('/home/luosy/data3382/image/*')
    train_mask_paths = glob('/home/luosy/data3382/mask/*')
    val_img_paths = glob('/home/luosy/data3382/val_image/*')
    val_mask_paths = glob('/home/luosy/data3382/val_mask/*')

    #train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = \
     #   train_test_split(img_paths, mask_paths, test_size=0.3, random_state=39)
    print("train_num:%s"%str(len(train_img_paths)))
    print("val_num:%s"%str(len(val_img_paths)))

    # create model
    #换模型需要修改的地方
    print("=> creating model %s" %args.arch)
    if args.arch=="transunet":
        from transunet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
        from transunet.vit_seg_modeling import VisionTransformer as ViT_seg
        config_vit = CONFIGS_ViT_seg["R50-ViT-B_16"]
        config_vit.n_classes = 2
        config_vit.n_skip = 3
        model = ViT_seg(config_vit, img_size=336, num_classes=config_vit.n_classes)
    elif args.arch=="convtransnet":
        from net.unet_transformer import U_Net_DSPP_Transformer
        model = U_Net_DSPP_Transformer(get_type='global')
    elif args.arch=="qaunet":
        model = qau_net.QAU_Net(3, 2)
    else:
        model = unet.U_Net(args)

    # model = PraNet_densenet.PraNet()
    # model = PraNet_Res2Net.PraNet()
    # model = network.deeplabv3plus_resnet50(num_classes=2)
    # model = Unet.Baseline(3)
    model = model.cuda()
    # model.load_state_dict(torch.load("/root/autodl-tmp/ECANet/models/LiTS_Unet_lym/2022-04-09-02-02-50/epoch273-0.9821-0.8964_model.pth"))
    print(count_params(model))

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
    
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.4)

    train_dataset = Dataset(args, train_img_paths, train_mask_paths, args.aug)
    val_dataset = Dataset(args, val_img_paths, val_mask_paths,val = True)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,90,150], gamma=0.3)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[38], gamma=0.3)
    # scheduler = PolyLR(optimizer,max_iters = 200,power=0.8)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min', factor=0.3, patience=10)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=False)

    log = pd.DataFrame(index=[], columns=[
        'epoch', 'lr', 'loss', 'iou','dice_1', 'dice_2', 'val_loss', 'val_iou','val_dice_1', 'val_dice_2'
    ])

    best_loss = 100
    best_iou = 0
    trigger = 0
    first_time = time.time()
    lr_list = []

    for epoch in range(args.epochs):
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        lr_list.append(lr)
        print('Epoch [%d/%d] | lr = [%f]' % (epoch, args.epochs, lr))
        # train for one epoch
        train_log = train(args, train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        val_log = validate(args, val_loader, model, criterion)

        print('loss %.4f - iou %.4f - dice_1 %.4f - dice_2 %.4f - val_loss %.4f - val_iou %.4f - val_dice_1 %.4f - val_dice_2 %.4f'
                  %(train_log['loss'], train_log['iou'], train_log['dice_1'], train_log['dice_2'], val_log['loss'], val_log['iou'], val_log['dice_1'], val_log['dice_2']))

        # print('loss %.4f - iou %.4f - dice %.4f ' %(train_log['loss'], train_log['iou'], train_log['dice']))
        end_time = time.time()
        print("time:", (end_time - first_time) / 60)

        tmp = pd.Series([
            epoch,
            args.lr,
            train_log['loss'],
            train_log['iou'],
            train_log['dice_1'],
            train_log['dice_2'],
            val_log['loss'],
            val_log['iou'],
            val_log['dice_1'],
            val_log['dice_2'],
        ], index=['epoch', 'lr', 'loss', 'iou', 'dice_1' ,'dice_2' ,'val_loss', 'val_iou', 'val_dice_1' ,'val_dice_2'])

        log = log.append(tmp, ignore_index=True)
        log.to_csv('models/{}/{}/log.csv'.format(args.name,timestamp), index=False)

        trigger += 1

        train_loss = train_log['loss']
        scheduler.step(train_loss)
        if train_loss < best_loss or epoch==150 or epoch==180:
            torch.save(model.state_dict(), 'models/{}/{}/epoch{}-{:.4f}-{:.4f}_model.pth'.format(args.name,timestamp,epoch,train_log['dice_1'],train_log['dice_2']))
            best_loss = train_loss
            print("=> saved best model")
            trigger = 0

       

        torch.cuda.empty_cache()
    # 绘图
    # plt.plot(range(len(lr_list)),lr_list)
    # plt.show()
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,2'
    main()
