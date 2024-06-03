# !/usr/bin/python3
# -*- coding: utf-8 -*-

###################################################################
## File: minfo.py
## Author: MiaoMiaoYang
## Created: 19.11.11
## Last Changed: 19.11.12
## Description: get basic info of image & label
###################################################################

## 得到原始图像的信息：长高宽、最大值、最小值
def getImageInfo(image_dire, file):
    '''
    得到一组数据的宽高切片数，以及图片的最大最小值
    :param image_dire: 图像文件所在的文件夹
    :param file: 写入.csv文件的文件路径
    :return: None
    '''

    from mtool.mio import get_files_name, get_medical_image
    from tqdm import tqdm
    import csv
    import os

    headers = ['Length', 'Width', 'Slice', 'Max', 'Min', 'file']

    files = get_files_name(image_dire)
    with open(file, 'w') as f:
        fcsv = csv.writer(f)
        fcsv.writerow(headers)
        index = 0
        for file in tqdm(files):
            if index>10:
                break
            index+=1

            image_path = os.path.join(image_dire, file)
            image, _, _, _, _ = get_medical_image(image_path)
            fcsv.writerow([image.shape[1], image.shape[2], image.shape[0],
                           image.max(), image.min(), file])


## 得到标签的信息：label所标识区域的长高宽、label区域中的最大值、最小值
def getLabelInfo(image_dire, label_dire, file):
    '''
    得到一组数据label中的信息
    label所标识区域的长高宽，label中的最大值、最小值
    :param image_dire: 图像文件所在的文件夹
    :param label_dire: 标签文件所在的文件夹
    :param file: 写入.csv文件的文件路径
    :return: None
    '''

    from mtool.mio import get_files_name, get_medical_image
    from mtool.mpretreat import getBounding
    from mtool.mutils import norm_zero_one
    from tqdm import tqdm
    import csv
    import os

    headers = ['Length', 'Width', 'Slice', 'Max', 'Min', 'Pixels', 'file']

    files = get_files_name(image_dire)
    with open(file, 'w') as f:
        fcsv = csv.writer(f)
        fcsv.writerow(headers)

        for file in tqdm(files):
            image_path = os.path.join(image_dire, file)
            label_path = os.path.join(label_dire, file)
            # label_path = os.path.join(label_dire, os.path.splitext(file)[0]+'.nii')
            image, _, _, _, _ = get_medical_image(image_path)
            imax, imin = image.max(), image.min()
            image = norm_zero_one(image)
            label, _, _, _, _ = get_medical_image(label_path)
            area = image * label
            area[area > 0] = area[area > 0] * (imax - imin) + imin
            bounding = getBounding(label)
            fcsv.writerow([bounding[1][1], bounding[2][1], bounding[0][1], area.max(), area.min(), label.sum(), file])


## 得到图像的SimpleITK信息
def getSitkInfo(image_dire, npy):
    '''
    得到图像的sitk信息
    :param image_dire: 图像文件夹
    :param npy: 保存信息的文件 .npy 文件
    :return:
    '''
    from mtool.mio import get_files_name, get_medical_image
    import numpy as np
    import os

    info = {}
    files = get_files_name(dire=image_dire)
    for file in files:
        path = os.path.join(image_dire, file)
        _, origin, spacing, direction, image_type = get_medical_image(path)
        info[os.path.splitext(file)[0]] = {'origin': origin, 'spacing': spacing,
                                           'direction': direction, 'image_type': image_type}
    info = np.asarray(info)
    np.save(file=npy, arr=info)


if __name__ == '__main__':
    # getLWS_MaxMin(image_dire='./data/imagesTr/')
    # getImageInfo(image_dire='./data/pretreat/Mask/', file='./mask.csv')
    getLabelInfo(image_dire='./data/imagesTr/', label_dire='./data/organ_label/', file='./organ.csv')
