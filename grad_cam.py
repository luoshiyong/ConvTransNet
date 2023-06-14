import warnings
from torchvision.models.segmentation import deeplabv3_resnet50
import torch
import torch.functional as F
import numpy as np
import requests
import torchvision
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam import GradCAM
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import logging
import network
import argparse
from torch.backends import cudnn
from net import unet_transformer

# 此处读取数据集的描述文件，如没有也可以自己写一个，添加本代码中需要的参数就可以了

cudnn.benchmark = False
# 此处添加需要得到CAM的文件名
img_name_list = ["backgroud", "liver", "tumor"]


def get_net():
    """
    Get Network for evaluation
    """
    model = unet_transformer.U_Net_DSPP_Transformer(get_type="mix")
    model.load_state_dict(torch.load("U:/paper3/ourformer233.pth"))
    model.eval()
    # for name, module in model.named_modules():
    #     print("name = {} | module = {}".format(name,module))
    return model



input123 = np.load("U:/paper3testimg/test_image/95_218.npy") # [1,3,336,336]
print("np.max = ",np.max(input123))
input = input123.transpose((2, 0, 1))
input = input.astype("float32")
input = torch.from_numpy(input).unsqueeze(0)


model = get_net()

if torch.cuda.is_available():
    model = model.cuda()
    input = input.cuda()


class SegmentationModelOutputWrapper(torch.nn.Module):
    def __init__(self, model):
        super(SegmentationModelOutputWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


model = SegmentationModelOutputWrapper(model)
output = model(input)
print("out shape = ",output.shape)
# [1,2,336,336]
normalized_masks = torch.sigmoid(output).cpu()
# 此处添加类名
sem_classes = [
    'background', 'liver', 'tumor'
]
sem_class_to_idx = {cls: idx  for (idx, cls) in enumerate(sem_classes)}

# 将需要进行CAM的类名写至此处
plaque_category = sem_class_to_idx["tumor"] # 2
liver_mask = normalized_masks[0, 0, :, :].detach().cpu().numpy()  # tumor
tumor_mask = normalized_masks[0, 1, :, :].detach().cpu().numpy()  # tumor
liver_mask[liver_mask > 0.5] = 1
liver_mask[liver_mask <= 0.5] = 0
tumor_mask[tumor_mask > 0.5] = 1
tumor_mask[tumor_mask <= 0.5] = 0
liver_mask_uint8 = 255 * np.uint8(liver_mask == 1)
liver_mask_float = np.float32(liver_mask == 1)
tumor_mask_uint8 = 255 * np.uint8(tumor_mask == 1)
tumor_mask_float = np.float32(tumor_mask == 1)




class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        return (model_output[self.category, :, :] * self.mask).sum()


# 此处修改希望得到特征图所在的网络层

target_layers = [model.model.third_stage3.fusion_block.conv2]
targets = [SemanticSegmentationTarget(1, tumor_mask_float)]

with GradCAM(model=model,
             target_layers=target_layers,
             use_cuda=torch.cuda.is_available()) as cam:
    grayscale_cam = cam(input_tensor=input,
                        targets=targets)[0, :]
    cam_image = show_cam_on_image(input123.astype("float32"), grayscale_cam, use_rgb=True)

img = Image.fromarray(cam_image)
# 保存位置
img.save("U:/paper3/paper_pipeline/testcam.png")

