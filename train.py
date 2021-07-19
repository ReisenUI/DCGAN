from __future__ import division
from __future__ import print_function

import os
import time
import torch
import torchvision.utils as vutils
import model
import record

from Utils import utils_image


class TrainModel(object):
    def __init__(self, dataloader, opt, device):
        super(TrainModel, self).__init__()
        self.opt = opt
        self.device = device
        self.train_model = self._get_train_model_(opt)
        record._record_dict_(self.opt, self.train_model["opt"])
        self.opt = self.train_model["opt"]
        self.dataloader = dataloader

    def _get_train_model_(self, opt):
        train_model = model._init_train_model_(opt, self.device)
        train_model = self._load_train_model_(train_model)
        return train_model

    def _save_train_model_(self):
        model_dict = model._get_model_dict_(self.train_model)
        file_full_path = record._get_check_point_file_full_path_(self.opt)
        torch.save(model_dict, file_full_path)

    def _load_train_model_(self, train_model):
        file_full_path = record._get_check_point_file_full_path_(self.opt)
        if os.path.exists(file_full_path) and self.opt.load_check:
            checkpoint = torch.load(file_full_path)
            train_model = model._load_model_dict_(train_model, checkpoint)
        return train_model

    def _train_step_(self, data):
        netG = self.train_model["netG"]
        netD = self.train_model["netD"]

        optimizerG = self.train_model["optimizerG"]
        optimizerD = self.train_model["optimizerD"]

        adversarial_loss = self.train_model["adversarial_loss"]
        device = self.device

        real_data = data[0].to(device)

        noise = model._get_noise_(real_data, self.opt, self.device)
        fake_data = netG(noise)
        label = model._get_label_(real_data, self.opt, self.device)

        errD, D_x, D_G_z1 = model._get_Discriminator_loss_(netD, optimizerD, real_data, fake_data.detach(), label, adversarial_loss, self.opt)
        errG, D_G_z2 = model._get_Generator_loss_(netG, netD, optimizerG, fake_data, label, adversarial_loss, self.opt)
        return errD, errG, D_x, D_G_z1, D_G_z2

    def _train_a_step_(self, data, i, epoch):
        start = time.time()
        errD, errG, D_x, D_G_z1, D_G_z2 = self._train_step_(data)
        end = time.time()
        step_time = end - start

        self.train_model["take_time"] = self.train_model["take_time"] + step_time

        print_every = 200
        if i % print_every == 0:
            record._print_status_(step_time * print_every,
                                  self.train_model["take_time"],
                                  epoch,
                                  i,
                                  errD,
                                  errG,
                                  D_x,
                                  D_G_z1,
                                  D_G_z2,
                                  self.opt,
                                  self.dataloader)
        return errD, errG

    def _DCGAN_eval_(self):
        fixed_noise = self.train_model["fixed_noise"]
        with torch.no_grad():
            netG = self.train_model["netG"]
            fake = netG(fixed_noise).detach().cpu()
            return fake

    def _save_generator_images_(self, iters, epoch, i):
        num_epochs = self.opt.n_epochs
        save_every = 500
        img_list = self.train_model["img_list"]

        if (iters % save_every == 0) or ((epoch == num_epochs - 1) and (i == len(self.dataloader) - 1)):
            fake = self._DCGAN_eval_()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            self._save_train_model_()

    def _train_iters_(self):
        num_epochs = self.opt.n_epochs
        G_losses = self.train_model["G_losses"]
        D_losses = self.train_model["D_losses"]
        iters = self.train_model["current_iters"]
        start_epoch = self.train_model["current_epoch"]

        for epoch in range(start_epoch, num_epochs):
            self.train_model["current_epoch"] = epoch
            for i, data in enumerate(self.dataloader, 0):
                errD, errG = self._train_a_step_(data, i, epoch)
                G_losses.append(errG.item())
                D_losses.append(errD.item())
                iters += 1
                self.train_model["current_iters"] = iters
                self._save_generator_images_(iters, epoch, i)

    def _DCGAN_testG_(self):
        fixed_noise = model._get_fixed_noise_(self.opt, self.device)
        with torch.no_grad():
            netG = self.train_model["netG"]
            fake = netG(fixed_noise).detach().cpu()
            return vutils.make_grid(fake, padding=2, normalize=True)

    def train(self):
        self._train_iters_()
        record.show_images(self.train_model, self.opt, self.dataloader, self.device)
