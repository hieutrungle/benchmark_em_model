import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import argparse
import utils
import logger
import models.efficientnet as efficientnet
from torchinfo import summary
import torchvision.transforms as transforms
import os
import data_io
import torch.optim as optim
import training
from torch.utils.data.dataloader import default_collate
import timer

import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import functools

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


def run_ipu(args):
    import poptorch
    import popdist

    popdist.init()

    # create model
    kwargs = {"device": args.device}
    model = efficientnet.efficientnet_prediction_model(num_classes=1, **kwargs)

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

    cache_dir = utils.mkdir_if_not_exist("./tmp")
    training_opts = ipu_training_options(
        gradient_accumulation=args.gradient_accumulation,
        replication_factor=args.replication_factor,
        device_iterations=args.device_iterations,
        number_of_ipus=args.num_ipus,
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

    layers_per_ipu = [8]
    if args.num_ipus == 1:
        layers_per_ipu = [8]
    elif args.num_ipus == 2:
        layers_per_ipu = [6, 2]
    elif args.num_ipus == 4:
        layers_per_ipu = [5, 1, 1, 1]
    elif args.num_ipus == 8:
        layers_per_ipu = [1, 1, 1, 1, 1, 1, 1, 1]
    ipu_config = {"layers_per_ipu": layers_per_ipu}
    model.parallelize(ipu_config).train()
    model = poptorch.trainingModel(model, options=training_opts, optimizer=optimizer)

    utils.mkdir_if_not_exist(args.model_path)

    # Train & Evaluate
    trainer = training.TorchTrainer(
        model, train_loader, test_loader, optimizer, DEVICE, args
    )
    trainer.train_ipu(args.epochs)


def get_dataset(args):
    world_size = dist.get_world_size()
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

    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, shuffle=True, drop_last=True
    )
    test_sampler = DistributedSampler(
        test_ds, num_replicas=world_size, shuffle=False, drop_last=False
    )
    batch_size = int(args.batch_size / float(world_size))
    logger.log(f"Batch size: {batch_size}")
    logger.log(f"World size: {world_size}")
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        sampler=train_sampler,
        batch_size=batch_size,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        sampler=test_sampler,
        batch_size=batch_size,
    )

    return train_loader, test_loader


def run_cuda(args, rank, world_size):
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    device = torch.device("cuda", rank)
    torch.backends.cudnn.benchmark = True
    train_loader, test_loader = get_dataset(args)

    # create model
    kwargs = {"device": args.device}
    model = efficientnet.efficientnet_prediction_model(num_classes=1, **kwargs)
    model.to(device)
    # use if model contains batchnorm.
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.compile(model, fullgraph=True)
    model = DDP(model, device_ids=[rank], output_device=rank)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.weight_decay,
        amsgrad=False,
    )

    # Train & Evaluate
    utils.mkdir_if_not_exist(args.model_path)
    trainer = training.TorchTrainer(
        model, train_loader, test_loader, optimizer, DEVICE, args
    )
    trainer.train(args.epochs)

    cleanup(rank)


def cleanup(rank):
    # dist.cleanup()
    dist.destroy_process_group()
    print(f"Rank {rank} is done.")


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_process(
    args,
    rank,  # rank of the process
    world_size,  # number of workers
    fn,  # function to be run
    # backend='gloo',# good for single node
    backend="nccl",  # the best for CUDA
    # backend="gloo",
):
    # information used for rank 0
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    dist.barrier()
    setup_for_distributed(rank == 0)
    fn(args, rank, world_size)


def main():
    torch.cuda.empty_cache()
    args = create_argparser().parse_args()
    logger.configure(dir="./logs")
    utils.log_args_and_device_info(args)

    if args.device == "ipu":
        logger.log("Using IPU and compiling model.")
        run_ipu(args)
        return
    else:
        logger.log("Using CUDA and compiling model.")
        world_size = torch.cuda.device_count()
        logger.log(f"World size: {world_size}")
        processes = []
        mp.set_start_method("spawn")

        for rank in range(world_size):
            p = mp.Process(target=init_process, args=(args, rank, world_size, run_cuda))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()


def create_argparser():
    """Parses command line arguments."""
    defaults = dict(
        data_dir="./data/256/images/25/256_1/train",
        test_dir="./data/256/images/25/256_1/test",
        model_path="./saved_models/061",
        verbose=True,
        batch_size=64,
        epochs=3,
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
        device_iterations=10,
        replication_factor=1,
        gradient_accumulation=1,
        num_ipus=1,
    )
    parser = argparse.ArgumentParser()
    utils.add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
