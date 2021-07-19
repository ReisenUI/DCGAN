from __future__ import division
from __future__ import print_function

import random
import torch
import torch.nn as nn
import torch.optim as optim

from DCGAN_architecture import Discriminator, Generator

import record


def _weights_init_(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def _random_init_():
    # manualSeed = 998
    manual_seed = random.randint(1, 10000)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)


def _get_a_net_(Net, opt, device):
    net = Net(opt).to(device)
    net.apply(_weights_init_)
    record._save_status_(opt, net)
    return net


def _get_optimizer_(net, opt):
    lr = opt.lr
    beta1 = opt.b1
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(beta1, 0.999))
    return optimizer


def _get_fixed_noise_(opt, device):
    latent_z = opt.latent_z
    fixed_noise = torch.randn(64, latent_z, 1, 1, device=device)
    return fixed_noise


def _init_train_model_(opt, device):
    _random_init_()
    netG = _get_a_net_(Generator, opt, device)
    netD = _get_a_net_(Discriminator, opt, device)
    adversarial_loss = nn.BCELoss()
    optimizerD = _get_optimizer_(netD, opt)
    optimizerG = _get_optimizer_(netG, opt)

    fixed_noise = _get_fixed_noise_(opt, device)

    train_model = {}
    train_model["netG"] = netG
    train_model["netD"] = netD
    train_model["adversarial_loss"] = adversarial_loss
    train_model["optimizerD"] = optimizerD
    train_model["optimizerG"] = optimizerG
    train_model["fixed_noise"] = fixed_noise

    train_model["G_losses"] = []
    train_model["D_losses"] = []
    train_model["current_iters"] = 0
    train_model["current_epoch"] = 0
    train_model["img_list"] = []
    train_model["opt"] = opt
    train_model["take_time"] = 0.0

    return train_model


def _run_Discriminator_(netD, data, label, loss):
    output = netD(data).view(-1)
    err = loss(output, label)
    err.backward()
    m = output.mean().item()
    return err, m


def _get_Discriminator_loss_(netD, optimizerD, real_data, fake_data, label, ad_loss, opt):
    netD.zero_grad()
    errD_real, D_x = _run_Discriminator_(netD, real_data, label, ad_loss)
    label.fill_(opt.fake_label)
    errD_fake, D_G_z1 = _run_Discriminator_(netD, fake_data, label, ad_loss)
    errD = errD_real + errD_fake
    optimizerD.step()
    return errD, D_x, D_G_z1


def _get_Generator_loss_(netG, netD, optimizerG, fake_data, label, ad_loss, opt):
    netG.zero_grad()
    label.fill_(opt.real_label)
    errG, D_G_z2 = _run_Discriminator_(netD, fake_data, label, ad_loss)
    optimizerG.step()
    return errG, D_G_z2


def _get_label_(data, opt, device):
    b_size = data.size(0)
    real_label = opt.real_label
    label = torch.full((b_size, ), real_label, dtype=torch.float, device=device)
    return label


def _get_noise_(data, opt, device):
    b_size = data.size(0)
    latent_z = opt.latent_z
    noise = torch.randn(b_size, latent_z, 1, 1, device=device)
    return noise


def _load_model_dict_(train_model, checkpoint):
    train_model["netG"].load_state_dict(checkpoint["netG"])
    train_model["netD"].load_state_dict(checkpoint["netD"])
    train_model["adversarial_loss"].load_state_dict(checkpoint["adversarial_loss"])
    train_model["optimizerD"].load_state_dict(checkpoint["optimizerD"])
    train_model["optimizerG"].load_state_dict(checkpoint["optimizerG"])

    train_model["fixed_noise"] = checkpoint["fixed_noise"]
    train_model["G_losses"] = checkpoint["G_losses"]
    train_model["D_losses"] = checkpoint["D_losses"]
    train_model["img_list"] = checkpoint["img_list"]
    train_model["current_iters"] = checkpoint["current_iters"]
    train_model["current_epoch"] = checkpoint["current_epoch"]
    train_model["opt"] = checkpoint["opt"]
    train_model["take_time"] = checkpoint["take_time"]
    return train_model


def _get_model_dict_(train_model):
    model_dict = {}
    model_dict["netG"] = train_model["netG"].state_dict()
    model_dict["netD"] = train_model["netD"].state_dict()
    model_dict["adversarial_loss"] = train_model["adversarial_loss"].state_dict()
    model_dict["optimizerD"] = train_model["optimizerD"].state_dict()
    model_dict["optimizerG"] = train_model["optimizerG"].state_dict()

    model_dict["fixed_noise"] = train_model["fixed_noise"]
    model_dict["G_losses"] = train_model["G_losses"]
    model_dict["D_losses"] = train_model["D_losses"]
    model_dict["img_list"] = train_model["img_list"]
    model_dict["current_iters"] = train_model["current_iters"]
    model_dict["current_epoch"] = train_model["current_epoch"]
    model_dict["opt"] = train_model["opt"]
    model_dict["take_time"] = train_model["take_time"]
    return model_dict
