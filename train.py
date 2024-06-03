import argparse
import types
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import Dataset
from loss import Loss
from model.network import Unet, weight_init_kaiming
from mtool.mio import get_json
from trainer import Trainer

DEBUG = False


def get_args():
    parser = argparse.ArgumentParser()

    # Training
    parser.add_argument("-t", "--tag", type=str, default="0104_single_pulp", help="Training Tag")
    parser.add_argument("-b", "--batch", type=int, default=15, help="Training Batch Size")
    parser.add_argument("-e", "--epoches", type=int, default=60, help="Training Epochs (Total)")
    parser.add_argument("-c", "--cuda", type=bool, default=True, help="Use CUDA ?")
    parser.add_argument("-f", "--folder", type=str, default="./k-folder.json",
                        help="K folder cross validation files json")
    parser.add_argument("-k", "--K", type=int, default=0,
                        help="K folder cross validation (start:0)",)
    parser.add_argument("-lr", "--lr", type=float, default=1e-5, help="Learning Rate")
    parser.add_argument("-pat", "--patience", type=int, default=10, help="Scheduler Patience")
    parser.add_argument("-pa", "--parallel", type=bool, default=True, help="Parallel or not")
    parser.add_argument("-cp", "--cpdire", type=str, default="./checkpoints", help="Checkpoint dire")
    parser.add_argument("-dcp", "--current_parameter", type=str, default=None, help="Go on using current parameters")

    # Data
    parser.add_argument("-id", "--image_dire", type=str, default="/deeptitan/eric/datasets/tooth/data/0104/single_pulp/image")
    parser.add_argument('-ld', "--tooth_dire", type=str, default='/deeptitan/eric/datasets/tooth/data/0104/single_pulp/tooth')
    parser.add_argument("-ad", "--pulp_dire", type=str, default='/deeptitan/eric/datasets/tooth/data/0104/single_pulp/pulp')
    parser.add_argument("-aug", "--aug_scale", type=int, default=5)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    ## Training
    tag = args.tag
    lr = args.lr  ## learning rate
    patience = args.patience  ## patience
    batch_size = args.batch  ## batch size
    total_epoches = args.epoches
    USE_CUDA = args.cuda
    parallel = args.parallel
    dire_cp = args.cpdire
    current_parameter = args.current_parameter
    k_folder_files = args.folder
    K = args.K
    value_span = None

    ## Data
    image_dire = args.image_dire
    tooth_dire = args.tooth_dire
    pulp_dire = args.pulp_dire
    aug_scale = args.aug_scale

    print('-----------------------------------------------')
    print('----------------- Parameters  -----------------')
    print('-----------------------------------------------')
    print("Training tag:       {}".format(tag))
    print("Batch size:         {}".format(batch_size))
    print("Total epoches:      {}".format(total_epoches))
    print("Use CUDA:           {}".format(USE_CUDA and torch.cuda.is_available()))
    print("Learning rate:      {}".format(lr))
    print("Lr patience:        {}".format(patience))
    print("Parallel:           {}".format(parallel))
    print("Checkpoint dire:    {}".format(dire_cp))
    print("Current Parameter:  {}".format(current_parameter))
    print("Augmentation scale: {}".format(aug_scale))
    print("K folder cross validation, files json: {}".format(k_folder_files))
    print("K folder corss validation, folder(K):  {}".format(K))
    ########################################################################
    print("image dire:  {}".format(image_dire))
    print("tooth dire:   {}".format(tooth_dire))
    print("pulp dire:  {}".format(pulp_dire))

    ## model
    model = Unet(in_channel=1, out_channel=1, mid_channels=[64, 128, 256, 512])
    model = weight_init_kaiming(model)
    print('-----------------------------------------------')
    print('--------------- Model Structure  --------------')
    print('-----------------------------------------------')
    print(model)

    ## Optimization
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print('-----------------------------------------------')
    print('--------------- Training Optimizer ------------')
    print('-----------------------------------------------')
    print(optimizer)

    ## Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=patience, verbose=True)

    ## Criterion
    criterion = Loss()
    print('-----------------------------------------------')
    print('---------------- Loss function  ---------------')
    print('-----------------------------------------------')
    print(criterion.__name__ if isinstance(criterion, types.FunctionType) else criterion)

    ## data
    train_test_k_folder_files = get_json(k_folder_files)
    train_files = train_test_k_folder_files["{}".format(K)]["train"]
    test_files = train_test_k_folder_files["{}".format(K)]["test"]
    print('-----------------------------------------------')
    print('------------------ Data Lists  ----------------')
    print('-----------------------------------------------')
    print("Training files:\n{}".format(train_files))
    print("Validation files:\n{}".format(test_files))

    dataset = Dataset(
        images_dire=image_dire,
        label_dires=[pulp_dire],
        label_tags=['pulp'],
        value_span=value_span,
        train_files=train_files,
        aug_scale=aug_scale
    )
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    trainer = Trainer(model=model, optimizer=optimizer, scheduler=scheduler, criterion=criterion,
                      dataloader=dataloader, test_files=test_files, value_span=value_span,
                      image_dire=image_dire,
                      label_dires=[pulp_dire],
                      label_tags=['pulp'],
                      dire_checkpoint=dire_cp, USE_CUDA=USE_CUDA, Tag=tag)

    trainer.run(epochs=total_epoches, parallel=parallel, checkpoint=current_parameter)
