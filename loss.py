import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
from torch.autograd import Variable
from torch.nn import functional as F


def compute_sdf(img_gt, out_shape):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdf to [-1,1]
    """

    img_gt = img_gt.astype(np.uint8)
    normalized_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]):  # batch size
        posmask = img_gt[b].astype(np.bool)
        if posmask.any():
            negmask = ~posmask
            posdis = distance(posmask)
            negdis = distance(negmask)
            boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
            sdf = (negdis - np.min(negdis)) / (np.max(negdis) - np.min(negdis)) - (posdis - np.min(posdis)) / (
                    np.max(posdis) - np.min(posdis))
            sdf[boundary == 1] = 0
            normalized_sdf[b] = sdf
            # assert np.min(sdf) == -1.0, print(np.min(posdis), np.max(posdis), np.min(negdis), np.max(negdis))
            # assert np.max(sdf) ==  1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))

    return normalized_sdf


def compute_dtm(img_gt, out_shape, normalize=False, fg=False):
    """
    compute the distance transform map of foreground in binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the foreground Distance Map (SDM)
    dtm(x) = 0; x in segmentation boundary
             inf|x-y|; x in segmentation
    """

    fg_dtm = np.zeros(out_shape)

    for b in range(out_shape[0]):  # batch size
        posmask = img_gt[b].astype(np.bool)
        if not fg:
            if posmask.any():
                negmask = 1 - posmask
                posdis = distance(posmask)
                negdis = distance(negmask)
                boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                if normalize:
                    fg_dtm[b] = (negdis - np.min(negdis)) / (np.max(negdis) - np.min(negdis)) + (
                            posdis - np.min(posdis)) / (np.max(posdis) - np.min(posdis))
                else:
                    fg_dtm[b] = posdis + negdis
                fg_dtm[b][boundary == 1] = 0
        else:
            if posmask.any():
                posdis = distance(posmask)
                boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                if normalize:
                    fg_dtm[b] = (posdis - np.min(posdis)) / (np.max(posdis) - np.min(posdis))
                else:
                    fg_dtm[b] = posdis
                fg_dtm[b][boundary == 1] = 0

    return fg_dtm


def hd_loss(seg_soft, gt, gt_dtm=None, one_side=True, seg_dtm=None):
    """
    compute huasdorff distance loss for binary segmentation
    input: seg_soft: softmax results,  shape=(b,x,y,z)
           gt: ground truth, shape=(b,x,y,z)
           seg_dtm: segmentation distance transform map; shape=(b,x,y,z)
           gt_dtm: ground truth distance transform map; shape=(b,x,y,z)
    output: boundary_loss; sclar
    """

    delta_s = (seg_soft - gt.float()) ** 2
    g_dtm = gt_dtm ** 2
    dtm = g_dtm if one_side else g_dtm + seg_dtm ** 2
    multipled = torch.einsum('bxyz, bxyz->bxyz', delta_s, dtm)
    # hd_loss = multipled.sum()*1.0/(gt_dtm > 0).sum()
    hd_loss = multipled.mean()

    return hd_loss


def sdf_loss(net_output, gt_sdm):
    # print('net_output.shape, gt_sdm.shape', net_output.shape, gt_sdm.shape)
    # ([4, 1, 112, 112, 80])

    smooth = 1e-5
    # compute eq (4)
    intersect = torch.sum(net_output * gt_sdm)
    pd_sum = torch.sum(net_output ** 2)
    gt_sum = torch.sum(gt_sdm ** 2)
    L_product = (intersect + smooth) / (intersect + pd_sum + gt_sum + smooth)
    # print('L_product.shape', L_product.shape) (4,2)
    L_SDF = 1 / 3 - L_product + torch.norm(net_output - gt_sdm, 1) / torch.numel(net_output)

    return L_SDF


def boundary_loss(outputs_soft, gt_sdf):
    """
    compute boundary loss for binary segmentation
    input: outputs_soft: sigmoid results,  shape=(b,2,x,y,z)
           gt_sdf: sdf of ground truth (can be original or normalized sdf); shape=(b,2,x,y,z)
    output: boundary_loss; sclar
    """
    pc = outputs_soft[:, 1, ...]
    dc = gt_sdf[:, 1, ...]
    multipled = torch.einsum('bxyz, bxyz->bxyz', pc, dc)
    bd_loss = multipled.mean()

    return bd_loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        num = target.size(0)
        smooth = 1

        m1 = input.view(num, -1)
        m2 = target.view(num, -1)
        intersection = (m1 * m2)

        score = (2. * intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        loss = 1 - score.sum() / num
        return loss


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, output, label):
        # loss_tooth = F.binary_cross_entropy(output[:, 0, :, :].unsqueeze(1), label[0])
        # loss_pulp = F.binary_cross_entropy(output[:, 1, :, :].unsqueeze(1), label[1])
        loss_pulp = F.binary_cross_entropy(output[:, 0, :, :].unsqueeze(1), label[0])

        # loss = loss_tooth * 0.1 + loss_pulp
        return loss_pulp

    def _get_name(self):
        return "Lung loss: lung [bce] blood [bce] airway [bce]"


if __name__ == '__main__':
    from mtool.mio import get_medical_image

    predict, _, _, _, _ = get_medical_image('./0003.nii.gz')
    gt, _, _, _, _ = get_medical_image('./data/origin/label/0003.nii.gz')

    predict = predict[120].astype(np.float)[np.newaxis, np.newaxis, :, :]
    gt = gt[120].astype(np.float)[np.newaxis, np.newaxis, :, :]

    predict = torch.from_numpy(predict)
    gt = torch.from_numpy(gt)

    criterion = Loss()
    loss = criterion(predict, gt)
    print(loss)
