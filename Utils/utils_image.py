import torchvision.utils as vutils
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch

from IPython.display import HTML
from matplotlib.animation import ImageMagickWriter


def training_image_show(dataloader, device):
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("./images Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.show()
    return


def _plot_real_and_fake_image_(real_batch, device, img_list, save_path):
    plt.figure(figsize=(30, 30))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64],
                                             padding=5, normalize=True).cpu(),
                            (1, 2, 0)))

    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    name = "real_and_fake.jpg"
    full_path_name = "{}/{}".format(save_path, name)
    plt.savefig(full_path_name)


def _show_generator_images_(G_losses, D_losses, save_path):
    plt.figure(figsize=(40, 20))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iteration")
    plt.ylabel("Loss")
    plt.legend()

    name = "G_D_losses.jpg"
    full_path_name = "{}/{}".format(save_path, name)
    plt.savefig(full_path_name)


def _show_img_list_(img_list):
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    HTML(ani.to_jshtml())
    plt.show()


def _save_img_list_(img_list, save_path, opt):
    metadata = dict(title="generator images", artist="Matplotlib", comment="Movie support!")
    writer = ImageMagickWriter(fps=1, metadata=metadata)
    ims = [np.transpose(i, (1, 2, 0)) for i in img_list]
    fig, ax = plt.subplots()
    with writer.saving(fig, "{}/img_list.gif".format(save_path), 500):
        for i in range(len(ims)):
            ax.imshow(ims[i])
            ax.set_title("step {}".format(i * 500))
            writer.grab_frame()


