import os
import numpy as np
import SimpleITK as sitk
# import nibabel as nib
from skimage import measure
from scipy.ndimage import label
import scipy.ndimage as ndi
import glob
from time import time

import copy
import math
import argparse
import random
import warnings
import datetime

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision
# import ttach as tta

from dataset.dataset import Dataset

# from net import Unet, unet_multi,unet_aspp
from utilities.utils import str2bool, count_params
import joblib
import imageio
from time import time

# import ttach as tta

test_ct_path = 'U:/paper3/40test/volume'   #需要预测的CT图像
seg_result_path = 'U:/paper3/40test/seg' #需要预测的CT图像标签，如果要在线提交codelab，需要先得到预测过的70例肝脏标签
pred_path = 'U:/paper3/pred/SANet_cfm_53'


"""test_ct_path = '/home/luosy/valdata/image'   #需要预测的CT图像
seg_result_path = '/home/luosy/valdata/mask' #需要预测的CT图像标签，如果要在线提交codelab，需要先得到预测过的70例肝脏标签

pred_path = '/home/luosy/valdata/predict'"""
np.random.seed(137)
if not os.path.exists(pred_path):
    os.mkdir(pred_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default=None,
                        help='')
    parser.add_argument('--training', type=bool, default=False,
                        help='whthere dropout or not')

    args = parser.parse_args()

    return args


def find_bb(volume):
    img_shape = volume.shape
    bb = np.zeros((6,), dtype=np.uint)
    bb_extend = 3
    # axis
    for i in range(img_shape[0]):
        img_slice_begin = volume[i, :, :]
        if np.sum(img_slice_begin) > 0:
            bb[0] = np.max([i - bb_extend, 0])
            break

    for i in range(img_shape[0]):
        img_slice_end = volume[img_shape[0] - 1 - i, :, :]
        if np.sum(img_slice_end) > 0:
            bb[1] = np.min([img_shape[0] - 1 - i + bb_extend, img_shape[0] - 1])
            break
    # seg
    for i in range(img_shape[1]):
        img_slice_begin = volume[:, i, :]
        if np.sum(img_slice_begin) > 0:
            bb[2] = np.max([i - bb_extend, 0])
            break

    for i in range(img_shape[1]):
        img_slice_end = volume[:, img_shape[1] - 1 - i, :]
        if np.sum(img_slice_end) > 0:
            bb[3] = np.min([img_shape[1] - 1 - i + bb_extend, img_shape[1] - 1])
            break

    # coronal
    for i in range(img_shape[2]):
        img_slice_begin = volume[:, :, i]
        if np.sum(img_slice_begin) > 0:
            bb[4] = np.max([i - bb_extend, 0])
            break

    for i in range(img_shape[2]):
        img_slice_end = volume[:, :, img_shape[2] - 1 - i]
        if np.sum(img_slice_end) > 0:
            bb[5] = np.min([img_shape[2] - 1 - i + bb_extend, img_shape[2] - 1])
            break

    return bb


def main():
    val_args = parse_args()
    # /home/luosy/TTNet/models/LiTS_Unet_lym/2022-05-08-14-36-20
    args = joblib.load('models/LiTS_Unet_lym/ttmp/args.pkl')
    print('Config -----')
    for arg in vars(args):
        print('%s: %s' % (arg, getattr(args, arg)))
    print('------------')
    joblib.dump(args, 'models/LiTS_Unet_lym/ttmp/args.pkl')
    # U:\TTNet\models\2021-12-20-15-54-01
    # create model
    print("=> creating model %s" % args.arch)

    # model = unet_multi.DBANet(3)
    # model = unet_aspp.U_Net("")
    # from net.unet53 import U_Net53
    # model = U_Net53("")
    from net.DBANet53 import DBANet
    model = DBANet(3)
    model = model.cuda()
    # C:\Users\luosy\Desktop\1080u\2022codeday
    model.load_state_dict(torch.load('U:/paper3/epoch16-0.9654-0.7568_model.pth'))
    model.eval()
    # model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode='mean')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

    for file_index, file in enumerate(os.listdir(test_ct_path)):
        start = time()

        # if file.replace('volume', 'segmentation').replace('nii','nii.gz') in os.listdir(pred_path):
        #     print('already predict {}'.format(file))
        #     continue
        # 将CT读入内存
        ct = sitk.ReadImage(os.path.join(test_ct_path, file), sitk.sitkInt16)
        ct_array = sitk.GetArrayFromImage(ct)

        mask = sitk.ReadImage(os.path.join(seg_result_path, file.replace('volume', 'segmentation')), sitk.sitkUInt8)
        mask_array = sitk.GetArrayFromImage(mask)

        mask_array[mask_array > 0] = 1

        print('start predict file:', file)
        print("raw shape = ", mask_array.shape)
        # ct_array[ct_array > 200] = 200
        # ct_array[ct_array < -200] = -200

        # ct_array = ct_array.astype(np.float32)
        # ct_array = ct_array / 200

        ct_array[ct_array > 240] = 240
        ct_array[ct_array < -100] = -100
        ct_array = ct_array.astype(np.float32)
        ct_array = (ct_array + 100) / 340
        bb = find_bb(mask_array)
        print("bb = ", bb)
        s1 = abs(bb[3] - bb[2])
        s2 = abs(bb[5] - bb[4])
        off = np.random.randint(10, 20)
        print("off = ", off)
        if s1 > s2:
            dp = (s1 - s2) // 2
            dp = min(dp + off, bb[4])
            print("dp = ", dp)
            bb[4] = bb[4] - dp
            bb[5] = min(bb[4] + s1 + 2 * off, 512)
            bb[2] = max(bb[2] - off, 0)
            bb[3] = min(bb[2] + s1 + 2 * off, 512)
        else:
            dp = (s2 - s1) // 2
            dp = min(dp + off, bb[2])
            print("dp = ", dp)
            bb[2] = bb[2] - dp
            bb[3] = min(bb[2] + s2 + 2 * off, 512)
            bb[4] = max(0, bb[4] - off)
            bb[5] = min(bb[4] + s2 + 2 * off, 512)

        # 找到肝脏区域开始和结束的slice，并各向外扩张slice
        start_slice = int(max(0, bb[0] - 1))
        end_slice = int(min(mask_array.shape[0] - 1, bb[1] + 2))
        # print("start_slice= {} end_slice+1= {} bb[2] = {} bb[3] ={} bb[4] ={} bb[5] = {}".format(start_slice,end_slice+1,bb[2],bb[3],bb[4],bb[5]))
        print("ct array shape = ",ct_array.shape)
        ct_crop = ct_array[start_slice :end_slice + 1, bb[2]:bb[3], bb[4]:bb[5]]
        ct_crop111 = ct_array[start_slice :end_slice + 1, bb[2]:bb[3], bb[4]:bb[5]]
        print("ct_crop start ={} |end ={}".format(start_slice,end_slice))
        # mask_crop = mask_array[start_slice+1:end_slice-1,bb[2]:bb[3],bb[4]:bb[5]]
        # print("start = {} | end = {}".format(start_slice,end_slice))
        ct_crop = ndi.zoom(ct_crop, (1, 336 / ct_crop.shape[1], 336 / ct_crop.shape[2]), order=3)
        # mask_crop = ndi.zoom(mask_crop, (1, 336 / mask_crop.shape[1], 336 / mask_crop.shape[2]), order=0)   #
        slice_predictions_l = np.zeros((ct_array.shape[0], 336, 336), dtype=np.int16)
        liver_pred = {i+start_slice+1:[] for i in range(0, ct_crop.shape[0] - 2)}
        d = sorted(list(liver_pred.items()), key=lambda x: x[0])

        print("maxidx = {} | minidx = {}".format(d[0],d[-1]))
        tumor_pred = {i+start_slice+1:[] for i in range(0, ct_crop.shape[0] - 2)}
        # pred_all = np.zeros((ct_array.shape[0], 336, 336), dtype=np.int16)
        slice_predictions = np.zeros((ct_array.shape[0], 512, 512), dtype=np.int16)
        infer_time = 0
        process_time = 0
        with torch.no_grad():
            for n_slice in range(0, ct_crop.shape[0] - 4):
                # print("start = {} | end ={} |pred_start = {} | pred_end ={}".format(
                #     n_slice+start_slice,n_slice+start_slice+4,n_slice + start_slice + 1,n_slice+start_slice+3
                # ))
                t1 = time()
                ct_tensor = torch.FloatTensor(ct_crop[n_slice: n_slice + 5]).cuda()
                ct_tensor = ct_tensor.unsqueeze(dim=0)
                # print('ct_tensor',ct_tensor.shape,n_slice)
                output = model(ct_tensor)
                output = torch.sigmoid(output).data.cpu().numpy()  # [bs,6,336,336]
                t2 = time()
                infer_time += t2 - t1
                # compute liver result
                liver1 = output[0,0,:,:]    # n_slice+start_slice+1
                liver2 = output[0,2,:,:]    # n_slice+start_slice+2
                liver3 = output[0,4, :, :]  # n_slice+start_slice+3
                tumor1 = output[0,1, :, :]
                tumor2 = output[0,3, :, :]
                tumor3 = output[0,5, :, :]
                liver_pred[n_slice+start_slice+1].append(liver1)
                liver_pred[n_slice + start_slice + 2].append(liver2)
                liver_pred[n_slice + start_slice + 3].append(liver3)
                tumor_pred[n_slice + start_slice + 1].append(tumor1)
                tumor_pred[n_slice + start_slice + 2].append(tumor2)
                tumor_pred[n_slice + start_slice + 3].append(tumor3)
                t3 = time()
                process_time += t3 - t2
            liver_pred = {key:np.mean(liver_pred[key],axis=0) for key in liver_pred.keys()}
            tumor_pred = {key:np.mean(tumor_pred[key], axis=0) for key in tumor_pred.keys()}
            for key in liver_pred.keys():
                probability_map = np.zeros([336,336],dtype = np.uint8)
                probability_map[liver_pred[key]>0.5] = 1
                probability_map[tumor_pred[key] > 0.5] = 2
                slice_predictions_l[key, :, :] = probability_map
            out_slice = ndi.zoom(slice_predictions_l, (
            1, ct_crop111.shape[1] / slice_predictions_l.shape[1], ct_crop111.shape[2] / slice_predictions_l.shape[2]),
                                 order=0)
            print("out_slice shape = ", out_slice.shape)
            slice_predictions[:, bb[2]:bb[3], bb[4]:bb[5]] = out_slice
            print("infer_time = {} | process_time = {}".format(infer_time, process_time))
            #
            pred_seg = slice_predictions
            pred_seg = pred_seg.astype(np.uint8)

            pred_seg = sitk.GetImageFromArray(pred_seg)

            pred_seg.SetDirection(ct.GetDirection())
            pred_seg.SetOrigin(ct.GetOrigin())
            pred_seg.SetSpacing(ct.GetSpacing())

            sitk.WriteImage(pred_seg, os.path.join(pred_path, file.replace('volume', 'segmentation')))

            speed = time() - start

            print(file, 'this case use {:.3f} s'.format(speed))
            print('-----------------------')

            torch.cuda.empty_cache()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()

