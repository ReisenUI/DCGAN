import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--load_check", type=bool, default=False, help="Choose whether to train(False) or test(True)")

parser.add_argument("--n_epochs", type=int, default=15, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=128, help="Batchsize during training")
parser.add_argument("--workers", type=int, default=2, help="Number of workers for dataloader")
parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate for optimizers")
parser.add_argument("--b1", type=float, default=0.5, help="Beta1 hyperparameter for Adam optimizers")
parser.add_argument("--n_cpu", type=int, default=8, help="Number of cpu threads to use during batch generation")
parser.add_argument("--n_gpu", type=int, default=1, help="Number of gpu to run on")
parser.add_argument("--latent_z", type=int, default=100, help="Size of z latent vector(size of generator input)")
parser.add_argument("--img_size", type=int, default=64, help="Spatial size of training images. All images"
                                                             " will be resized to this size using a transformer")
parser.add_argument("--channels", type=int, default=3, help="Number of channels in the training images.(Color images)")
parser.add_argument("--G_feature", type=int, default=64, help="Number of Generator feature")
parser.add_argument("--D_feature", type=int, default=64, help="Number of Discriminator feature")
parser.add_argument("--fake_label", type=int, default=0, help="Fake label")
parser.add_argument("--real_label", type=int, default=1, help="Real label")

opt = parser.parse_known_args()[0]