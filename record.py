from __future__ import division
from __future__ import print_function

import etc
import os
from Utils import utils_image

output_path = "./images2/"
# cur_dataset = "img_align_celeba"
cur_dataset = "faces"

def _get_param_str_(opt):
    param_str = "{}_{}_{}_{}_{}_{}_{}".format(
        cur_dataset,
        opt.img_size,
        opt.batch_size,
        opt.G_feature,       # G feature
        opt.D_feature,       # D feature
        opt.latent_z,        # latent-Z
        opt.lr               # learning rate
    )
    return param_str


def _get_check_point_path_(opt):
    param_str = _get_param_str_(opt)
    directory = "{}/save/{}/".format(
        output_path + cur_dataset,
        param_str
    )
    os.makedirs(directory, exist_ok=True)
    return directory


def _get_check_point_file_full_path_(opt):
    path = _get_check_point_path_(opt)
    param_str = _get_param_str_(opt)
    file_full_path = "{}{}checkpoint.tar".format(
        path,
        param_str
    )
    return file_full_path


def _write_output_(opt, con):
    save_path = _get_check_point_path_(opt)
    file_full_path = "{}/output".format(save_path)
    f = open(file_full_path, "a")
    f.write("{}\n".format(con))
    f.close()


def _save_status_(opt, con):
    print(con)
    _write_output_(opt, con)


def _print_status_(step_time, take_time, epoch, i, errD, errG, D_x, D_G_z1, D_G_z2, opt, dataloader):
    num_epochs = opt.n_epochs
    print_str = "[{}/{}]\t[{}/{}]\t Loss_D:{}\t Loss_G:{}\t D(x):{}\t D(G(z)):{}/{}\t take_time:{}".format(
        epoch,
        num_epochs,
        i,
        len(dataloader),
        errD.item(),
        errG.item(),
        D_x,
        D_G_z1,
        D_G_z2,
        take_time
    )
    _save_status_(opt, print_str)


def _record_dict_(opt, dic):
    _save_status_(opt, "config:")
    d = vars(dic)
    for key in d:
        dic_str = "{} : {}".format(key, d[key])
        _save_status_(opt, dic_str)


def show_images(train_model, opt, dataloader, device):
    G_losses = train_model["G_losses"]
    D_losses = train_model["D_losses"]
    img_list = train_model["img_list"]
    save_path = _get_check_point_path_(opt)

    utils_image._show_generator_images_(G_losses, D_losses, save_path)
    utils_image._save_img_list_(img_list, save_path, opt)
    real_batch = next(iter(dataloader))
    utils_image._plot_real_and_fake_image_(real_batch, device, img_list, save_path)
