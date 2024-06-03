import os
import time
import traceback
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from model.network import Unet
from mtool.meval import get_dice
from mtool.mio import get_medical_image, save_medical_image
from mtool.mutils.mutils import norm_zero_one
from dataset import get_rotate, get_rotate1

DEBUG = False


class Trainer(object):
    def __init__(self, model, optimizer=None, scheduler=None, criterion=None,
                 dataloader=None, test_files=None, value_span=None,
                 image_dire=None, label_dires=None, label_tags=None,
                 dire_checkpoint='./checkpoints', USE_CUDA=False, Tag="Train"):

        ## Model
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.USE_CUDA = USE_CUDA
        self.dire_checkpoint = dire_checkpoint
        self.Tag = Tag

        ## Data
        self.dataloader = dataloader
        self.test_files = test_files
        self.value_span = value_span

        ## Training
        self.image_dire = image_dire
        self.label_dires = label_dires
        self.label_tags = label_tags

        self.loss = 0
        self.current_epoch = 0
        self.total_epochs = 0
        self.is_load_param = False
        self.parallel = False

    def run(self, epochs=200, parallel=False, checkpoint=None):
        if checkpoint is not None:
            self.load_param(checkpoint)

        self.parallel = parallel
        if parallel:
            self.model = nn.DataParallel(self.model)

        if self.USE_CUDA and torch.cuda.is_available():
            self.model = self.model.cuda()

        self.eval()
        self.total_epochs = epochs
        for i in range(1, epochs + 1):
            self.current_epoch = i
            if DEBUG:
                self.train()
                self.save_param()
                self.eval()
                self.update()
            else:
                try:
                    self.train()
                    self.save_param()
                    self.eval()
                    self.update()
                except Exception as e:
                    print(e)
                    traceback.print_exc()
                    self.save_param()
                    break

    def train(self):
        print('-----------------------------------------------')
        print('----------------  Training ... ----------------')
        print('-----------------------------------------------')
        print('Training {} Starting epoch {}/{}.'.format(self.Tag, self.current_epoch, self.total_epochs))
        time.sleep(0.5)

        self.model.train()

        sum_losses = 0.
        threshold_grad = 1e4
        for index, train_data in tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
            images = train_data[0].float()    # train_data[0] is image
            labels = [data.float() for data in train_data[1:]]    # train_data[1] tooth; train_data[2] pulp;
            if self.USE_CUDA and torch.cuda.is_available():
                images = images.cuda()
                labels = [label.cuda() for label in labels]

            predictions = self.model(images)
            loss = self.criterion(predictions, labels)
            sum_losses += float(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), threshold_grad)
            self.optimizer.step()

        time.sleep(0.5)
        print('{} Epoch{} finished ! Loss: {}'.format(self.Tag, self.current_epoch, sum_losses))


    def predict(self, images):

        images = norm_zero_one(images, span=self.value_span)
        shape = images.shape[0]

        # 失状面  冠状面
        # images = get_rotate(images)

        # 换用另一个面要用
        # images = get_rotate1(images)


        images = np.asarray(images)

        results = {tag: [] for tag in self.label_tags}
        for image in images:
            image = np.asarray(image[np.newaxis, np.newaxis, :, :]).astype(np.float)
            image = torch.from_numpy(image).float()

            if self.USE_CUDA and torch.cuda.is_available():
                image = image.cuda()
            predictions = self.model(image)

            if self.USE_CUDA and torch.cuda.is_available():
                predictions = predictions.cpu()
            predictions = predictions.numpy()[0]

            for index, tag in enumerate(self.label_tags):
                results[tag].append(predictions[index])

        for tag in results:
            result = np.asarray(np.asarray(results[tag]).astype(np.float) > 0.5).astype(np.int)
            # print(result.shape)

            # 矢状面  冠状面
            # result = result.transpose([1, 0, 2])[(256 - shape) // 2: (256 - shape) // 2 + shape, :, :]

            # get_rotate1要用
            # result = result.transpose([0, 2, 1])

            # print(result.shape)
            results[tag] = result

        return results

    def predict_file(self, src_file, dst_files):
        assert len(dst_files) == len(self.label_tags), "dst_files doesn't matched the dst tags"

        self.model.eval()
        if self.USE_CUDA and torch.cuda.is_available():
            self.model = self.model.cuda()

        print("predict src file: {}".format(src_file))
        with torch.no_grad():
            image, origin, spacing, direction, image_type = get_medical_image(src_file)
            image = np.asarray(image).astype(np.float)

            predictions = self.predict(image)
            for index, tag in enumerate(predictions):
                result = np.asarray(predictions[tag]).astype(np.int)
                # result = get_largest_n_connected_region(result, 1)[0]
                save_medical_image(result, dst_files[index], origin, spacing, direction, image_type)
                print("tag:{} dst file:{}".format(tag, dst_files[index]))

        print("file:{} predict finished!".format(src_file))
        print('-------------------------------------------')

    def eval(self):

        print('-----------------------------------------------')
        print('----------------  Evaluation ------------------')
        print('-----------------------------------------------')
        self.model.eval()

        indexes_dice = {tag: [] for tag in self.label_tags}
        with torch.no_grad():
            for index, file in enumerate(self.test_files):

                ## load image
                image, _, _, _, _ = get_medical_image(os.path.join(self.image_dire, file))

                ## predict results
                predictions = self.predict(image)

                ## load labels
                eval_str = "file:{:22s} ".format(file)
                for label_tag, label_dire in zip(self.label_tags, self.label_dires):
                    label, _, _, _, _ = get_medical_image(os.path.join(label_dire, file))
                    label = np.asarray(label > 0.5).astype(np.int)

                    result = predictions[label_tag]
                    dice = get_dice(result, label)
                    indexes_dice[label_tag].append(dice)
                    eval_str += "{} dice: {:.4f} ".format(label_tag, dice)
                print(eval_str)

        indexes_dice = {tag: np.asarray(indexes_dice[tag]).mean() for tag in self.label_tags}
        self.loss = sum([indexes_dice[tag] for tag in self.label_tags])
        print("Total: ", *["{} dice:{:.4f}  ".format(tag, indexes_dice[tag]) for tag in self.label_tags])

    def update(self):
        ## 学习率下降
        self.scheduler.step(self.loss)

    def save_param(self, ):
        print('-----------------------------------------------')
        print('------------  Saving Checkpoints  -------------')
        print('-----------------------------------------------')
        os.makedirs(self.dire_checkpoint, exist_ok=True)
        torch.save(self.model.module.state_dict() if self.parallel else self.model.state_dict(),
                   os.path.join(self.dire_checkpoint,
                                "Tag-{}-Epoch-{}-cp-{}.pth".format(self.Tag, self.current_epoch,
                                                                   time.strftime("%Y-%m-%d-%H-%M-%S",
                                                                                 time.localtime()))))
        print("save checkpoint done!")

    def load_param(self, checkpoint="."):
        print('-----------------------------------------------')
        print('-----------  Loading Checkpoints  -------------')
        print('-----------------------------------------------')
        self.model.load_state_dict(torch.load(checkpoint))
        print("Checkpoint: {} loaded !".format(checkpoint))


def test_Trainer():
    ###########################################
    ## Trainer.eval
    ###########################################
    from dataset import Dataset
    from mtool.mio import get_files_name
    from torch.utils.data import DataLoader
    import torch.optim as optim
    from loss import Loss

    batch_size = 5
    radius = 3
    lr = 1e-4
    patience = 10

    ## model
    model = Unet().cuda()

    ## optimization
    optimizer = optim.Adam(model.parameters(), lr=lr)

    ## schedular
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience, verbose=True)

    ## criterion
    criterion = Loss()
    files = get_files_name(dire='./data/origin/image')
    dataset = Dataset()
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, drop_last=True, shuffle=True)

    image_dire = './data/origin/image'
    label_dire = './data/origin/label'
    test_files = get_files_name(image_dire)

    trainer = Trainer(model=model, optimizer=optimizer, scheduler=scheduler, criterion=criterion,
                      dataloader=dataloader, test_files=test_files,
                      image_dire=image_dire, dire_checkpoint='./checkpoints', USE_CUDA=True, Tag='covid-19')
    # trainer.eval()
    trainer.run(epochs=10)

    pass


if __name__ == '__main__':
    test_Trainer()
