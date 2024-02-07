import os
import shutil
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt


def mkdir_with_clear(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


def mkdir_if_not_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)


def display_sample(sample, i, title=""):
    image_processed = sample.cpu().permute(0, 2, 3, 1)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.numpy().astype(np.uint8)

    image_pil = PIL.Image.fromarray(image_processed[0])
    # plt.figure(figsize=(10, 10))
    image_dir = "./images"
    mkdir_if_not_exists(image_dir)
    image_pil.save(image_dir / f"{title}_image_at_step_{i}.png")
    # plt.close()


def save_images(
    images,
    num_prompt,
    num_images_per_prompt,
    n_ipu,
    inference_replication_factor,
    prompt,
):
    num_images = len(images)

    fig, axes = plt.subplots(num_prompt, num_images_per_prompt, constrained_layout=True)
    # fig.set_size_inches(9, 9)

    for i, image in enumerate(images):
        if num_prompt == 1 and num_images_per_prompt == 1:
            ax = axes
        elif num_prompt == 1:
            ax = axes[i % num_images_per_prompt]
        else:
            ax = axes[i // num_images_per_prompt, i % num_images_per_prompt]
        ax.imshow(image)
        ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect("equal")

    fig.suptitle(f"prompt: {prompt}")
    # fig.tight_layout()
    # fig.subplots_adjust(wspace=0, hspace=0)

    current_dir = os.path.dirname(os.path.realpath(__file__))
    image_dir = os.path.join(current_dir, "images")
    mkdir_if_not_exists(image_dir)
    fig.savefig(
        os.path.join(
            image_dir,
            f"n_ipu_{n_ipu}_num_prompt_{num_prompt}_num_images_per_prompt_{num_images_per_prompt}_inference_replication_factor_{inference_replication_factor}.png",
        ),
        dpi=150,
    )
    plt.close()
