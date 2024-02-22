import os
import torch.distributed as dist


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
    fn(rank, world_size)
