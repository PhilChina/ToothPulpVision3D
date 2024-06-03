import os
import random
import time
from datetime import datetime

import numpy as np
from scipy.ndimage import rotate, shift
from torch.utils.data import Dataset as dataset
from tqdm import tqdm

from mtool.mio import get_files_name, get_medical_image, save_json
from mtool.mutils.mtrain import get_k_folder_cross_validation_samples
from mtool.mutils.mutils import norm_zero_one

DEBUG = False


def get_rotate(image):
    blank = np.zeros(shape=[128, 256, 128])

    shape = image.shape[0]
    blank[:, (256 - shape) // 2: (256 - shape) // 2 + shape, :] = np.asarray(image).transpose([1, 0, 2])
    return blank

def get_rotate1(image):
    blank = np.zeros(shape=[128, 256, 128])
    shape = image.shape[0]
    blank[:, (256 - shape) // 2: (256 - shape) // 2 + shape, :] = np.asarray(image).transpose([2, 0, 1])
    return blank

def get_rotate_0_1(image, pad_value):
    blank = np.full(shape=[128, 256, 128], fill_value=pad_value)
    shape = image.shape[0]
    blank[:, (256 - shape) // 2: (256 - shape) // 2 + shape, :] = np.asarray(image).transpose([1, 0, 2])
    return blank


def get_rotate_0_2(image, pad_value):
    # blank = np.zeros(shape=[128, 256, 128])
    blank = np.full(shape=[128, 256, 128], fill_value=pad_value)
    shape = image.shape[0]
    blank[:, (256 - shape) // 2: (256 - shape) // 2 + shape, :] = np.asarray(image).transpose([2, 0, 1])
    return blank


def get_k_folder(image_dire, dst_file, K=0):
    files = get_files_name(dire=image_dire)
    results = get_k_folder_cross_validation_samples(files, K)
    save_json(results, file=dst_file)


class Dataset(dataset):
    def __init__(self, images_dire, label_dires, label_tags, train_files, aug_scale=1, value_span=None):
        super(Dataset, self).__init__()
        # if value_span is None:
        #     value_span = [200, 2500]
        print('-----------------------------------------------')
        print('----------- Loading Training Images -----------')
        print('-----------------------------------------------')
        print("Augment scale:     {}".format(aug_scale))
        # print("Dicom value range: {}".format(value_span))
        for tag, dire in zip(label_tags, label_dires):
            print("Label tag:{:10s} dire:{}".format(tag, dire))
        time.sleep(0.5)

        self.images_dire = images_dire
        self.label_dires = label_dires
        self.label_tags = label_tags
        self.aug_scale = aug_scale

        assert len(self.label_dires) == len(self.label_tags), \
            "the number of label dires doesn't match the number of its tags"

        self.images = []
        self.labels = {tag: [] for tag in self.label_tags}

        for index, file in tqdm(enumerate(train_files), total=len(train_files)):
            if index > 1 and DEBUG:
                continue

            # load medical image
            images, _, _, _, _ = get_medical_image(os.path.join(self.images_dire, file))
            images = norm_zero_one(images)

            for slice in images:
                self.images.append(slice)


            # images_rotate_0_1 = get_rotate_0_1(images, pad_value=images.min())
            # for slice in images_rotate_0_1:
            #     self.images.append(slice)
            #
            # images_rotate_0_2 = get_rotate_0_2(images, pad_value=images.min())
            # for slice in images_rotate_0_2:
            #     self.images.append(slice)

            # load different label
            for (tag, dire) in zip(self.label_tags, self.label_dires):
                label, _, _, _, _ = get_medical_image(os.path.join(dire, file))

                label = np.asarray(label > 0.5).astype(np.int)
                assert label.shape == images.shape, \
                    "the shape of images doesn't match the shape of label {}".format(tag)
                for slice in label:
                    self.labels[tag].append(slice)


                # label_rotate_0_1 = get_rotate_0_1(label, pad_value=0)
                # label_rotate_0_1 = np.asarray(label_rotate_0_1 > 0.5).astype(np.int)
                # assert label_rotate_0_1.shape == images_rotate_0_1.shape, \
                #     "the shape of images doesn't match the shape of label {}".format(tag)
                # for slice in label_rotate_0_1:
                #     self.labels[tag].append(slice)
                #
                # label_rotate_0_2 = get_rotate_0_2(label, pad_value=0)
                # label_rotate_0_2 = np.asarray(label_rotate_0_2 > 0.5).astype(np.int)
                # assert label_rotate_0_2.shape == images_rotate_0_2.shape, \
                #     "the shape of images doesn't match the shape of label {}".format(tag)
                # for slice in label_rotate_0_2:
                #     self.labels[tag].append(slice)

        time.sleep(0.5)
        self.imsize = len(self.images) * self.aug_scale
        print("Load finished! num:{}".format(self.imsize))

    def __getitem__(self, index):
        images = self.images[index // self.aug_scale]
        labels = [self.labels[tag][index // self.aug_scale] for tag in self.label_tags]

        if index % self.aug_scale != 0:
            random.seed(datetime.now())
            angle = random.uniform(-5, 5)
            images = rotate(images, angle, reshape=False)
            labels = [rotate(label, angle, reshape=False) for label in labels]

            shifts = [5, 5]
            x_shift = random.uniform(-shifts[0], shifts[0])
            y_shift = random.uniform(-shifts[1], shifts[1])
            images = shift(images, shift=[x_shift, y_shift])
            labels = [shift(label, shift=[x_shift, y_shift]) for label in labels]

            images = norm_zero_one(images, span=[0.0, 1.0])
            images = np.asarray(images)[np.newaxis, :, :].astype(np.float)
            labels = [np.asarray(label > 0.5).astype(np.float)[np.newaxis, :, :] for label in labels]
            return [images, *labels]

        images = np.asarray(images)[np.newaxis, :, :].astype(np.float)
        labels = [np.asarray(label > 0.5).astype(np.float)[np.newaxis, :, :] for label in labels]
        return [images, *labels]

    def __len__(self):
        return self.imsize


def test_Dataset():
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    files = get_files_name('E:\\tooth\\v2\\data\\single\\image')

    dataset = Dataset(
        images_dire='E:\\tooth\\v2\\data\\single\\image',
        label_dires=['E:\\tooth\\v2\\data\\single\\tooth',
                     'E:\\tooth\\v2\\data\\single\\pulp'],
        label_tags=['tooth', 'pulp'],
        value_span=None,
        train_files=files,
        aug_scale=2
    )

    dataloader = DataLoader(dataset=dataset, batch_size=2, shuffle=True, drop_last=True)
    for index, (image, tooth, pulp) in tqdm(enumerate(dataloader), total=len(dataloader)):
        image = image.numpy()[0]
        tooth = tooth.numpy()[0]
        pulp = pulp.numpy()[0]
        print(image.shape)
        plt.figure(22)
        plt.subplot(221)
        plt.imshow(image[0], cmap='gray')
        plt.subplot(222)
        plt.imshow(pulp[0], cmap='gray')
        plt.subplot(223)
        plt.imshow(tooth[0], cmap='gray')
        # plt.show()
        plt.savefig('./result/' + '{}.png'.format(index))


if __name__ == "__main__":
    # get_k_folder(image_dire='./data/single/image',K=3,dst_file='./k-folder.json')
    test_Dataset()
    pass

