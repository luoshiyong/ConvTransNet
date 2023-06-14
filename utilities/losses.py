import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

class Edge_loss(nn.Module):
    def __init__(self):
        super(Edge_loss, self).__init__()
        self.mse = torch.nn.MSELoss(reduction='mean')
    def forward(self, input, target):
        # input = torch.sigmoid(input)
        target = F.interpolate(target, scale_factor=0.5, mode='bilinear', align_corners=False)
        bce = F.binary_cross_entropy(input, target)
        mse = self.mse(input,target)
        return 0.5 * bce + 0.5*mse
"""
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()

    def forward(self, input, target):
        # input = torch.sigmoid(input)
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)

        input_1 = input[:,0,:,:]
        input_2 = input[:,1,:,:]
        target_1 = target[:,0,:,:]
        target_2 = target[:,1,:,:]

        input_1 = input_1.view(num, -1)
        input_2 = input_2.view(num, -1)

        target_1 = target_1.view(num, -1)
        target_2 = target_2.view(num, -1)

        intersection_1 = (input_1 * target_1)
        intersection_2 = (input_2 * target_2)

        dice_1 = (2. * intersection_1.sum(1) + smooth) / (input_1.sum(1) + target_1.sum(1) + smooth)
        dice_2 = (2. * intersection_2.sum(1) + smooth) / (input_2.sum(1) + target_2.sum(1) + smooth)

        dice_1 = 1 - dice_1.sum() / num
        dice_2 = 1 - dice_2.sum() / num

        dice = (dice_1+dice_2)/2.0
        return 0.5 * bce + dice
        return bce
"""
def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp

def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn, tn
class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.):
        """
        """
        super(SoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth

        dc = nominator / (denominator + 1e-8)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc
"""
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.dc1 = SoftDiceLoss()
        self.dc2 = SoftDiceLoss()
    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        input = torch.sigmoid(input)
        num = target.size(0)

        input_1 = input[:,0,:,:]
        input_2 = input[:,1,:,:]
        target_1 = target[:,0,:,:]
        target_2 = target[:,1,:,:]
        # print("input1 = {} | target1 = {} ".format(input_1.shape,target_1.shape))
        ce1  = self.dc1(input_1,target_1)
        ce2  = self.dc2(input_2,target_2)
        # ce2 = self.ce_loss2(input_2,target_2)
        return 2+bce + ce1 + ce2
"""
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()

    def forward(self, input, target):
        # input = torch.sigmoid(input)
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)

        input_1 = input[:,0,:,:]
        input_2 = input[:,1,:,:]
        target_1 = target[:,0,:,:]
        target_2 = target[:,1,:,:]

        input_1 = input_1.view(num, -1)
        input_2 = input_2.view(num, -1)

        target_1 = target_1.view(num, -1)
        target_2 = target_2.view(num, -1)

        intersection_1 = (input_1 * target_1)
        intersection_2 = (input_2 * target_2)

        dice_1 = (2. * intersection_1.sum(1) + smooth) / (input_1.sum(1) + target_1.sum(1) + smooth)
        dice_2 = (2. * intersection_2.sum(1) + smooth) / (input_2.sum(1) + target_2.sum(1) + smooth)

        dice_1 = 1 - dice_1.sum() / num
        dice_2 = 1 - dice_2.sum() / num

        dice = (dice_1+dice_2)/2.0
        return 0.5 * bce + dice

class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super(LovaszHingeLoss, self).__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss
