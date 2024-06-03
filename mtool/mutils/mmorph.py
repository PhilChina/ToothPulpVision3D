#!/usr/bin/python3
# -*- coding:utf-8 -*-

###################################################################
## File: mpretreat.py
## Author: MiaoMiaoYang
## Created: 20.08.22
## Last Changed: 20.08.22
## Description: morphological operation
###################################################################
import numpy as np


## 图像的数值截断
def truncate(image, min, max):
    '''
    :param image: 图像
    :param min:  最小值
    :param max:  最大值
    :return: image
    '''
    img = image.copy()
    img[img > max] = max
    img[img < min] = min
    return img


## 将数组二值化 - 大津
def binaryOTSU(array, fore=None):
    """
    将数组使用otsu进行二值化
    :param   array:  进行二值化的数组
    :param   fore:   前景的阈值，如果为None则不进行前景提取
    :return: binary: 得到的二值化数组
    """

    ## 计算阈值
    from skimage.filters import threshold_otsu

    ## 将前景提取来
    tmp = array[array > fore] if fore != None else array
    thresh = threshold_otsu(tmp)

    ## 二值化
    binary = array > thresh
    return binary


## 3D高斯模糊
def gaussianBlur3D(image, radius=1, sigema=1.5, gpu=True):
    '''
    三维高斯模糊，其中需要用到pytorch
    :param image: 三维图像
    :return:
    '''

    image = np.asarray(image)
    assert len(image.shape) == 3, "input image is not a 3-dim image"

    import math
    import torch
    import torch.nn.functional as F

    ## 高斯的计算公式
    def calc(x, y, z):
        res1 = 1 / ((math.sqrt(2 * math.pi) * sigema) ** 3)
        res2 = math.exp(-(x ** 2 + y ** 2 + z ** 2) / (2 * (sigema ** 2)))
        return res1 * res2

    ## 高斯核
    def gaussKernel():
        length = radius * 2 + 1
        result = np.zeros((length, length, length))
        for i in range(length):
            for j in range(length):
                for k in range(length):
                    result[i, j, k] = calc(i - radius, j - radius, k - radius)
        sum = result.sum()
        return result / sum

    ## 准备高斯核
    kernel = gaussKernel()
    kernel = kernel[np.newaxis, np.newaxis, :, :, :]
    kernel = torch.from_numpy(kernel)

    ## 准备图像
    image = image[np.newaxis, np.newaxis, :, :, :]
    image = torch.from_numpy(image).type_as(kernel)

    if gpu and torch.cuda.is_available():
        kernel = kernel.cuda()
        image = image.cuda()
    else:
        print('use cpu in gaussian')

    ## 进行卷积
    result = F.conv3d(input=image, weight=kernel, stride=1, padding=radius)
    if gpu and torch.cuda.is_available():
        result = result.cpu()
    result = result.numpy()[0][0]
    return result


## 二维数组进行左右翻转
def get_flip_lr(array):
    '''
    将二维数组进行左右翻转
    :param array: 二维数组
    :return: numpy数组
    '''
    array = np.asarray(array)
    assert len(array.shape) == 2, 'get_flip_lr in ./mtool/mutils.py doesn\'t get 2D array'
    return np.fliplr(array)


## 二维数组进行上下翻转
def get_flip_ud(array):
    '''
    将二维数组进行上下翻转
    :param array: 二维数组
    :return: numpy数组
    '''
    array = np.asarray(array)
    assert len(array.shape) == 2, 'get_flip_ud in ./mtool/mutils.py doesn\'t get 2D array'
    return np.flipud(array)


## 将圆形视野中的前景提出出来
def get_foreground(indire, outdire):
    '''
    将CT/MRI图中的前景提取出来，将保存好的文件以原格式储存到另一文件夹中
    因为一些医学图像的窗位是圆形的，E.g.圆形窗位中的背景是-1024左右，而圆形窗位之外的方框背景在-2048
    这个函数将所有的背景归一到-1024
    :param indire: 医学图像的文件夹
    :param outdire: 保存到的医学图像文件夹
    :return: None
    注意！此函数不可以使用.dcm格式数据
    '''

    import os
    assert "{} is not existed!".format(indire), os.path.exists(indire)
    assert "{} is not a directory!".format(indire), os.path.isdir(indire)

    ## 如果输出文件夹不存在，创建输出文件夹
    os.makedirs(outdire, exist_ok=True)

    from mtool.mio import get_files_name, get_medical_image, save_medical_image
    from scipy.ndimage.morphology import binary_fill_holes
    import numpy as np

    files = get_files_name(dire=indire)
    for file in files:
        image, origin, spacing, direction, image_type = get_medical_image(indire + file)
        output = []
        for img in image:
            ## 对每一层图片进行处理
            bin = binaryOTSU(array=img)
            bin = np.asarray(bin).astype(int)
            bin = binary_fill_holes(bin).astype(int)
            slice = bin * (img + 1024) - 1024
            output.append(slice)
        output = np.asarray(output)
        save_medical_image(array=output, origin=origin, spacing=spacing, direction=direction,
                           target_path=outdire + file, type=image_type)
        print("file:{} is done!".format(file))

    print("getForeground complete!")
