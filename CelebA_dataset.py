from __future__ import division
from __future__ import print_function

import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms


def cele_data_loader(opt):
    data_path = "./images2/"
    img_size = opt.img_size
    batch_size = opt.batch_size
    workers = opt.workers

    dataset = dset.ImageFolder(
        root=data_path,
        transform=transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=workers)

    return dataloader
