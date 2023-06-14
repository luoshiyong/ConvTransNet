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
from net import unet_transformer
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
from utilities.metrics import dice_coef,dice_coef_test, batch_iou, mean_iou, iou_score
import utilities.losses as losses
from utilities.utils import str2bool, count_params
import pandas as pd
from net import  unet
from utilities.lr_policy import PolyLR

# from  net.small_ecanet import PraNet121
# 换模型需要修改的地方
arch_names = list(unet.__dict__.keys())
loss_names = list(losses.__dict__.keys())
loss_names.append('BCEWithLogitsLoss')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')

    # 换模型需要修改的地方
    parser.add_argument('--arch', '-a', metavar='ARCH', default='Unet',
                        choices=arch_names,
                        help='model architecture: ' +
                             ' | '.join(arch_names) +
                             ' (default: NestedUNet)')
    # 换数据集需要修改的地方
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
    # 换模型需要修改的地方
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--early-stop', default=50, type=int,
                        metavar='N', help='early stopping (default: 30)')

    # 换模型需要修改的地方
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
    parser.add_argument('--checkpoint', default='U:/paper3/sanet_all_deformable2.pth',
                        help='image file extension')
    parser.add_argument('--model_name', default="unetplusplus",
                        help='dataset name')
    parser.add_argument('--save_path', default="U:/paper3testimg/pred_tmp/",
                        help='dataset name')
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





def validate(args, val_loader, model):
    ious = AverageMeter()
    dices_1s = AverageMeter()
    dices_2s = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target, filename) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            # compute output
            if args.deepsupervision:
                outputs = model(input)
                loss = 0
            else:
                output = model(input)
                iou = iou_score(output, target)
                dice_1 = dice_coef_test(output, target)[0]
                dice_2 = dice_coef_test(output, target)[1]
                if dice_1>0.98 and dice_2>0.95:
                    print("dice1={}|dice2={}*** ->{}".format(dice_1,dice_2,filename))
                output = torch.sigmoid(output)
                # print("input shape = ", input[0].shape)
                raw = input.detach().cpu().numpy()[0][1]
                # print("raw shape = ",raw.shape)
                pred_liver = output.detach().cpu().numpy()[0][0]
                pred_tumor = output.detach().cpu().numpy()[0][1]
                liver_gt =  target.detach().cpu().numpy()[0][0]
                tumor_gt =  target.detach().cpu().numpy()[0][1]
                pred_liver[pred_liver > 0.5] = 255
                pred_liver[pred_liver <= 0.5] = 0
                """
                pred_liver[pred_liver > 0.5] = 133
                pred_liver[pred_liver <= 0.5] = 0
                pred_tumor[pred_tumor > 0.5] = 133
                pred_tumor[pred_tumor <= 0.5] = 0
                pred_liver[pred_tumor > 1] = 255
                """
                liver_gt[liver_gt>0] = 133
                liver_gt[tumor_gt>0] = 255
                im = Image.fromarray(np.uint8(pred_liver))
                im1 = Image.fromarray(np.uint8(liver_gt))
                im.convert('L').save(args.save_path + filename[0] + '.jpg')
                # im1.convert('L').save("U:/paper3testimg/groundtruth/" + filename[0] + '.jpg')
                # rawout.convert('L').save("U:/paper3testimg/rawimg/" + filename[0] + '.jpg')
            ious.update(iou, input.size(0))
            dices_1s.update(torch.tensor(dice_1), input.size(0))
            dices_2s.update(torch.tensor(dice_2), input.size(0))

    log = OrderedDict([
        ('iou', ious.avg),
        ('dice_1', dices_1s.avg),
        ('dice_2', dices_2s.avg)
    ])
    print("log = ",log)
    return log


def main():
    args = parse_args()
    # args.dataset = "datasets"


    # Data loading code
    val_img_paths = glob('U:/paper3testimg/test_image/*')
    val_mask_paths = glob('U:/paper3testimg/test_mask/*')
    print("val_num:%s" % str(len(val_img_paths)))

    # create model
    # 换模型需要修改的地方
    print("=> creating model %s" % args.arch)

    # from net.DBANet import DBANet
    # model = DBANet(3)
    # from net.unet_transformer import U_Net_DSPP_Transformer
    # model = U_Net_DSPP_Transformer(get_type="global")
    # import network
    # model = network.deeplabv3plus_resnet50(num_classes=2,pretrained_backbone=False)
    # from net.DBANet_deformable import DBANet
    # model = DBANet(3)
    if args.model_name=='unet':
        model = unet.U_Net(args)
        model.load_state_dict(torch.load("U:/paper3/unet32_135.pth"))
    elif args.model_name=="our_transformer":
        model = unet_transformer.U_Net_DSPP_Transformer(get_type="mix")
        from net.lsyformer  import U_Net_DSPP_Transformer
        model = U_Net_DSPP_Transformer(get_type="mix")
    elif args.model_name=="qaunet":
        from net import qau_net
        model = qau_net.QAU_Net(3,2)
    elif args.model_name =="transunet":
        from transunet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
        from transunet.vit_seg_modeling import VisionTransformer as ViT_seg
        config_vit = CONFIGS_ViT_seg["R50-ViT-B_16"]
        config_vit.n_classes = 2
        config_vit.n_skip = 3
        model = ViT_seg(config_vit, img_size=336, num_classes=config_vit.n_classes)
    elif args.model_name =="unetplusplus":
        from net import unetplusplus
        model = unetplusplus.NestedUNet(2)
        model.load_state_dict(torch.load("U:/paper3/unetplus_66.pth"))
    elif args.model_name =="medical_transformer":
        import lib
        model = lib.models.axialnet.MedT(img_size=336,imgchan=3)
    elif args.model_name=="resunet":
        from net import resunet
        model = resunet.ResUnet(3)
    else:
        model = unet.U_Net(args)
    model = model.cuda()

    print(count_params(model))




    val_dataset = Dataset(args, val_img_paths, val_mask_paths, val=True)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,90,150], gamma=0.3)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[38], gamma=0.3)
    # scheduler = PolyLR(optimizer,max_iters = 200,power=0.8)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=False)


    best_loss = 100
    best_iou = 0
    trigger = 0
    first_time = time.time()
    lr_list = []


    val_log = validate(args, val_loader, model)

    print('val_iou %.4f - val_dice_1 %.4f - val_dice_2 %.4f'% (val_log['iou'], val_log['dice_1'], val_log['dice_2']))

    end_time = time.time()
    print("time:", (end_time - first_time) / 60)
    torch.cuda.empty_cache()



if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,2'
    main()
