"""
compute the param and the flops of different network

"""
import torch
from time import time
# from net import resunet
from thop import profile
import torch.nn as nn
# from net import qau_net
# from net import unet
# from net import unet_transformer
import os
from PIL import Image
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

# import lib
# model = lib.models.axialnet.MedT(img_size = 336, imgchan = 3)
# model = qau_net.QAU_Net(3,2)
# model = unet_transformer.U_Net_DSPP_Transformer(get_type="mix")
# model = unet.U_Net("")

"""from transunet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from transunet.vit_seg_modeling import VisionTransformer as ViT_seg
config_vit = CONFIGS_ViT_seg["R50-ViT-B_16"]
config_vit.n_classes = 2
config_vit.n_skip = 3
model = ViT_seg(config_vit, img_size=336, num_classes=config_vit.n_classes)"""
input = torch.randn(2,5,336,336)
#torch.save(model.state_dict(),'unet.pth')
# model = resunet.ResUnet(3)
# from net.unetplusplus import NestedUNet
# model = NestedUNet(2)
# from net.unet_transformer import U_Net_DSPP_Transformer
# model = U_Net_DSPP_Transformer(get_type='global')
# from utnet.utnet import UTNet
# model = UTNet(3, 32, 2, reduce_size=8, block_list='1234', num_blocks=[2,2,4,8], num_heads=[4,4,4,4], projection='interp', attn_drop=0.1, proj_drop=0.1, rel_pos=False, aux_loss=None, maxpool=True)
from net.DBANet_deformable import DBANet56
model = DBANet56(3)
start = time()
with torch.no_grad():
    out = model(input)
speed = time() - start
print('this case use {:.3f} s'.format(speed))
print(sum(p.numel() for p in model.parameters() if p.requires_grad))