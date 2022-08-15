from torch.nn import BatchNorm2d, LeakyReLU, Dropout, Sequential, Conv2d, ConvTranspose2d, ReLU
from torch.nn.init import normal_, constant_


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        normal_(m.weight.data, 1.0, 0.02)
        constant_(m.bias.data, 0)


def Deconv(n_input, n_output, k_size=4, stride=2, padding=1):
    return Sequential(
        ConvTranspose2d(
            n_input, n_output,
            kernel_size=k_size,
            stride=stride,
            padding=padding,
            bias=False),
        BatchNorm2d(n_output),
        ReLU(inplace=True))


def Conv(n_input, n_output, k_size=4, stride=2, padding=0, bn=False):
    return Sequential(
        Conv2d(
            n_input, n_output,
            kernel_size=k_size,
            stride=stride,
            padding=padding, bias=False),
        BatchNorm2d(n_output),
        LeakyReLU(0.2, inplace=True),
        Dropout(p=0.2, inplace=False))
