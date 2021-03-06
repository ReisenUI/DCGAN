from __future__ import division
from __future__ import print_function

import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import numpy as np


from etc import opt
import train
import model
import record
import CelebA_dataset
import matplotlib.pyplot as plt
from Utils import utils_image
from matplotlib.animation import ImageMagickWriter

# Change the path wherever your dataset is
output_path = "./images2/"

# Choose which device to run on ( GPU | CPU )
device = torch.device("cuda:0" if (torch.cuda.is_available() and opt.n_gpu > 0) else "cpu")

if __name__ == '__main__':
    os.makedirs(output_path, exist_ok=True)
    dataloader = CelebA_dataset.cele_data_loader(opt)

    # 展示部分训练集图片
    # utils_image.training_image_show(dataloader, device)

    g = train.TrainModel(dataloader, opt, device)
    if not opt.load_check:
        g._load_train_model_(opt)
        # print(g.train_model)
        g.train()
    elif opt.load_check:
        g._load_train_model_(opt)
        # utils_image._show_img_list_(g.train_model['img_list'])
        # utils_image._plot_real_and_fake_image_(next(iter(dataloader)), device, g.train_model['img_list'], record._get_check_point_path_(opt))
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(np.transpose(g._DCGAN_testG_(), (1, 2, 0)))
        plt.show()
        
