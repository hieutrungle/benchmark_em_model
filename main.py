import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import argparse
import utils
import logger
import models.efficientnet as efficientnet
from torchinfo import summary
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import data_io
import torch.optim as optim
import training
from torch.utils.data.dataloader import default_collate

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type != "cpu":
    NUM_GPUS = len([torch.cuda.device(i) for i in range(torch.cuda.device_count())])
    # NUM_GPUS = 1
else:
    NUM_GPUS = 0
os.environ["DEVICE"] = str(DEVICE.type)
os.environ["NUM_GPUS"] = str(NUM_GPUS)


def ipu_training_options(
    gradient_accumulation,
    replication_factor,
    device_iterations,
    number_of_ipus,
    cache_dir,
):
    import popart
    import poptorch

    opts = poptorch.Options()
    opts.randomSeed(12345)
    opts.deviceIterations(device_iterations)

    # Use Pipelined Execution
    opts.setExecutionStrategy(
        poptorch.PipelinedExecution(poptorch.AutoStage.AutoIncrement)
    )

    # Use Stochastic Rounding
    opts.Precision.enableStochasticRounding(True)

    # Half precision partials for matmuls and convolutions
    opts.Precision.setPartialsType(torch.float16)

    opts.replicationFactor(replication_factor)

    opts.Training.gradientAccumulation(gradient_accumulation)

    # Return the final result from IPU to host
    opts.outputMode(poptorch.OutputMode.Final)

    # Cache compiled executable to disk
    opts.enableExecutableCaching(cache_dir)

    # # Setting system specific options
    # # On-chip Replicated Tensor Sharding of Optimizer State
    # opts.TensorLocations.setOptimizerLocation(
    #     poptorch.TensorLocationSettings()
    #     # Optimizer state lives on IPU if running on a POD16
    #     .useOnChipStorage(number_of_ipus == 16)
    #     # Optimizer state sharded between replicas with zero-redundancy
    #     .useReplicatedTensorSharding(number_of_ipus == 16)
    # )

    # # Available Transient Memory For matmuls and convolutions operations dependent on system type
    # if number_of_ipus == 16:
    #     amps = [0.08, 0.28, 0.32, 0.32, 0.36, 0.38, 0.4, 0.1]
    # else:
    #     amps = [0.15, 0.18, 0.2, 0.25]

    # opts.setAvailableMemoryProportion({f"IPU{i}": mp for i, mp in enumerate(amps)})

    ## Advanced performance options ##

    # Only stream needed tensors back to host
    opts._Popart.set("disableGradAccumulationTensorStreams", True)

    # Copy inputs and outputs as they are needed
    opts._Popart.set(
        "subgraphCopyingStrategy", int(popart.SubgraphCopyingStrategy.JustInTime)
    )

    # Parallelize optimizer step update
    opts._Popart.set(
        "accumulateOuterFragmentSettings.schedule",
        int(popart.AccumulateOuterFragmentSchedule.OverlapMemoryOptimized),
    )
    opts._Popart.set("accumulateOuterFragmentSettings.excludedVirtualGraphs", ["0"])

    # Limit number of sub-graphs that are outlined (to preserve memory)
    opts._Popart.set("outlineThreshold", 10.0)

    # Only attach to IPUs after compilation has completed.
    opts.connectionType(poptorch.ConnectionType.OnDemand)
    return opts


def main():
    torch.cuda.empty_cache()
    args = create_argparser().parse_args()
    logger.configure(dir="./logs")
    utils.log_args_and_device_info(args)

    torch.multiprocessing.set_start_method("spawn")

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
    if DEVICE.type == "cuda" and NUM_GPUS > 0:
        model = model.to("cuda")
    else:
        model = model.to("cpu")
    # model = model.to("cuda")  # put model to device (GPU)

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

    test_dir = args.test_dir
    test_ds = data_io.ImageCurrentDataset(
        test_dir,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        ),
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.weight_decay,
        amsgrad=False,
    )
    train_loader = None
    test_loader = None
    training_opts = None
    if args.device == "ipu":
        import poptorch

        logger.log("Using IPU.")
        cache_dir = utils.mkdir_if_not_exist("./tmp")
        training_opts = ipu_training_options(
            gradient_accumulation=1,
            replication_factor=1,
            device_iterations=1,
            number_of_ipus=1,
            cache_dir=cache_dir,
        )
        train_loader = poptorch.DataLoader(
            options=training_opts,
            dataset=train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
        )
        test_loader = poptorch.DataLoader(
            options=training_opts,
            dataset=test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
        )

        model = poptorch.trainingModel(
            model, options=training_opts, optimizer=optimizer
        )
    else:
        logger.log("Using CUDA and compiling model.")
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=16,
            pin_memory=True,
            drop_last=True,
        )
        test_loader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=16,
            pin_memory=True,
            drop_last=False,
        )
        model = torch.compile(model)

    utils.mkdir_if_not_exist(args.model_path)

    # Train & Evaluate
    trainer = training.TorchTrainer(
        model, train_loader, test_loader, optimizer, DEVICE, args
    )
    trainer.train(args.epochs)


def create_argparser():
    """Parses command line arguments."""
    defaults = dict(
        data_dir="./data/061/train",
        test_dir="./data/061/test",
        model_path="./saved_models/061",
        verbose=True,
        batch_size=64,
        epochs=1,
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
        device="cuda",
    )
    parser = argparse.ArgumentParser()
    utils.add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
