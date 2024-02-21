import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import time
import shutil
from pathlib import Path
import glob
import errno
import argparse
import utils
import logger
import models.efficientnet as efficientnet
from torchinfo import summary
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import sys
import train_module
import data_io
import matplotlib.pyplot as plt

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# NUM_GPUS = len([torch.cuda.device(i) for i in range(torch.cuda.device_count())])

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
if DEVICE.type != "cpu":
    NUM_GPUS = len([torch.cuda.device(i) for i in range(torch.cuda.device_count())])
    # NUM_GPUS = 1
else:
    NUM_GPUS = 0
os.environ["DEVICE"] = str(DEVICE.type)
os.environ["NUM_GPUS"] = str(NUM_GPUS)


def main():
    torch.cuda.empty_cache()
    args = create_argparser().parse_args()
    logger.configure(dir="./logs")
    utils.log_args_and_device_info(args)

    # Model Initialization
    # model = efficientnet.efficientnet_model(num_classes=10)

    # model = efficientnet.efficientnet_prediction_model(num_classes=1)

    # weight_path = "/home/xubuntu/research/EfficientNetV2/saved_models/061_1/checkpoints/sst-epoch=000-val_loss=0.01345.pt"
    # create model
    model = efficientnet.efficientnet_prediction_model(num_classes=1)
    # load weight
    # checkpoint = torch.load(weight_path, map_location=DEVICE)
    # model.load_state_dict(checkpoint)

    # input shape is of 2 dimensions, however, the model expects 4 dimensions
    # input_shape = (1, 3, 400, 400)
    input_shape = (1, 3, 256, 256)
    summary(
        model,
        input_shape,
        depth=2,  # go into 2 sub layers depth
        col_names=(
            "input_size",
            "output_size",
            "num_params",
        ),
        row_settings=("depth", "ascii_only"),
    )
    model = model.to(torch.device(DEVICE))  # put model to device (GPU)

    # Data Preparation
    # normalize = transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    # )
    train_dir = args.data_dir
    train_ds = data_io.ImageCurrentDataset(
        train_dir,
        transform=transforms.Compose(
            [   
                
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
            ]
        ),
    )
    # for i, (image, current) in enumerate(train_ds):
    #     # plot the image in the batch, along with the corresponding labels
    #     print(f"index: {i}; image: {image}; current: {current.shape}")
    #     fig = plt.figure(figsize=(16, 8))
    #     plt.imshow(image.permute(1, 2, 0))
    #     plt.title(f"Current: {current}")
    #     plt.show()
    #     break
    # sys.exit()
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4 * NUM_GPUS,
        pin_memory=True,
        drop_last=True,
    )

    test_dir = args.test_dir
    test_ds = data_io.ImageCurrentDataset(
        test_dir,
        transform=transforms.Compose(
            [
                
                transforms.ToTensor(),
            ]
        ),
    )
    # for i, (image, current) in enumerate(test_ds):
    #     # plot the image in the batch, along with the corresponding labels
    #     print(f"index: {i}; image: {image.shape}; current: {current.shape}")
    #     fig = plt.figure(figsize=(16, 8))
    #     plt.imshow(image.permute(1, 2, 0))
    #     plt.title(f"Current: {current}")
    #     plt.show()
    #     break
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4 * NUM_GPUS,
        pin_memory=True,
        drop_last=False,
    )

    # sys.exit()

    # Data Preparation
    # normalize = transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    # )
    # traindir = os.path.join(args.data_dir, "train")
    # traindir = os.path.join(args.data_dir)
    # traindir2 = os.path.join(args.data_dir2) if args.data_dir2 != "None" else None
    # valdir = os.path.join("./data/imgs", "val")
    # train_dataset = data_io.CombinedImageDataset(
    #     traindir,
    #     traindir2,
    # transforms.Compose(
    #     [
    #         transforms.RandomResizedCrop(64),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         # normalize,
    #     ]
    # ),
    # )
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=4 * NUM_GPUS,
    #     pin_memory=True,
    #     drop_last=True,
    # )
    # val_dataset = datasets.ImageFolder(
    #     valdir,
    #     transforms.Compose(
    #         [
    #             transforms.Resize(64),
    #             transforms.CenterCrop(64),
    #             transforms.ToTensor(),
    #             # normalize,
    #         ]
    #     ),
    # )
    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=4 * NUM_GPUS,
    #     pin_memory=True,
    #     drop_last=False,
    # )

    # Train & Evaluate
    train_module.train(model, train_loader, test_loader, args)


def create_argparser():
    """Parses command line arguments."""
    defaults = dict(
        data_dir="../Data4CNN/56/train",
        test_dir="../Data4CNN/56/test",
        model_path="./saved_models/56_256",
        verbose=True,
        batch_size=64,
        epochs=100,
        # lr=1e-4,
        
        lr=0.001,
        warm_up_portion=0.2,
        # weight_decay=1e-6,
        weight_decay=0,
        momentum=0.9,
        log_interval=10,
        resume="",
        iter=-1,  # -1 means resume from the best model
        conductivity=1,
    )
    parser = argparse.ArgumentParser()
    utils.add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
