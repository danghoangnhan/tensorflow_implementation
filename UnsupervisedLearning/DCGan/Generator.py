from torch import nn
from torch.nn import ConvTranspose2d, Tanh

from SupervisedLearning.DCGan import Deconv


class Generator(nn.Module):
    def __init__(self, z=100, nc=64):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            Deconv(z, nc * 8, 4, 1, 0),
            Deconv(nc * 8, nc * 4, 4, 2, 1),
            Deconv(nc * 4, nc * 2, 4, 2, 1),
            Deconv(nc * 2, nc, 4, 2, 1),
            ConvTranspose2d(nc, 3, 4, 2, 1, bias=False),
            Tanh()
        )

    def forward(self, input):
        return self.net(input)
