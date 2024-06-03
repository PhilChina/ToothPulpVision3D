import os

from tqdm import tqdm
from model.network import Unet
from mtool.mio import get_json
from trainer import Trainer

if __name__ == '__main__':
    ###########################################
    # Trainer.eval
    ###########################################

    batch_size = 5
    radius = 3
    lr = 1e-4
    patience = 10
    K = 0

    # model
    model = Unet(in_channel=1, out_channel=1, mid_channels=[64, 128, 256, 512]).cuda()
    trainer = Trainer(model=model, optimizer=None, scheduler=None, criterion=None, dataloader=None, test_files=None,
                      value_span=None,
                      image_dire=None, label_dires=None, label_tags=['pulp'], dire_checkpoint='./checkpoints',
                      USE_CUDA=True, Tag='predict')
    path = './checkpoints/Tag-0104_single_pulp-Epoch-7-cp-2021-01-06-12-18-48.pth'
    trainer.load_param(path)

    # trainer.predict_file(src_file='./data/single/image/CHEN YAN LIAN-24.nrrd',
    #                      dst_files=[
    #                          './result/pulp/CHEN YAN LIAN-24.nrrd',
    #                          './result/tooth/CHEN YAN LIAN-24.nrrd'
    #                      ])
    # exit()

    k_folder_files = './k-folder.json'
    train_test_k_folder_files = get_json(k_folder_files)
    test_files = train_test_k_folder_files["{}".format(K)]["test"]

    # image_dire = '/deeptitan/eric/datasets/tooth/data/single/image/'
    image_dire = './data/0104/single/image/'
    dst_tooth_dire = './result/tooth/'

    dst_pulp_dire = './result/pulp/'

    os.makedirs(dst_tooth_dire, exist_ok=True)
    os.makedirs(dst_pulp_dire, exist_ok=True)

    for file in test_files:
        trainer.predict_file(src_file=os.path.join(image_dire, file), dst_files=[os.path.join(dst_pulp_dire, file)])
