# import diffusers
# import transformers
# from diffusers import (
#     AutoencoderKL,
#     UNet2DConditionModel,
#     PNDMScheduler,
# )
# from transformers import CLIPTextModel, CLIPTokenizer
import os
import utils
import logger
import torch
import argparse
import torchvision.models as models
import timer

try:
    import poptorch
except ImportError:
    print("poptorch not installed")


def main():
    args = create_argparser().parse_args()
    current_dir = os.path.dirname(os.path.realpath(__file__))
    log_dir = os.path.join(current_dir, "logs")
    utils.mkdir_if_not_exists(log_dir)
    logger.configure(dir=log_dir)
    prompt = "a shiba inu in a zen garden, acrylic painting"
    model_id = "stabilityai/stable-diffusion-2-1"

    model = models.resnet18().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # compiled_model = torch.compile(model, mode="max-autotune")
    compiled_model = torch.compile(model, mode="reduce-overhead")
    # compiled_model = model

    for i in range(3):
        with timer.Timer(logger.log):
            x = torch.randn(16, 3, 224, 224).cuda()
            optimizer.zero_grad()
            y = compiled_model(x)
            y.sum().backward()
            optimizer.step()

    # if args.device == "ipu":
    #     import ipu_benchmark

    #     ipu_benchmark.benchmark(prompt, model_id)
    # elif args.device == "gpu":
    #     import gpu_benchmark

    #     gpu_benchmark.benchmark(prompt, model_id)


def create_argparser() -> argparse.ArgumentParser:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", "-d", type=str, required=True)
    # parser.add_argument("--num_prompts", "-np", type=int, default=1)
    # parser.add_argument("--num_images_per_prompt", "-nip", type=int, default=1)
    # parser.add_argument("--num_devices", "-nd", type=int, default=1)
    # parser.add_argument("--replication_factor", "-r", type=int, default=1)
    parser.add_argument("--verbose", "-v", type=bool, default=False)
    return parser


if __name__ == "__main__":
    main()
