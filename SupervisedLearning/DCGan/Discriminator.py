from torch import nn
from torch.nn import Conv2d, Sigmoid, LeakyReLU, Sequential
from SupervisedLearning.DCGan import Conv


class Discriminator(nn.Module):
    def __init__(self, nc=64):
        super(Discriminator, self).__init__()
        self.net = Sequential(
            Conv2d(
                3, nc,
                kernel_size=4,
                stride=2,
                padding=1, bias=False),
            LeakyReLU(0.2, inplace=True),
            Conv(nc, nc * 2, 4, 2, 1),
            Conv(nc * 2, nc * 4, 4, 2, 1),
            Conv(nc * 4, nc * 8, 4, 2, 1),
            Conv2d(nc * 8, 1, 4, 1, 0, bias=False),
            Sigmoid())

    def forward(self, input):
        return self.net(input)


def disc_1():
    return nn.Sequential(
        # Input is 3 x 256 x 256
        nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(16),
        nn.LeakyReLU(0.3, inplace=True),
        # Layer Output: 16 x 128 x 128

        nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(0.3, inplace=True),
        # Layer Output: 32 x 64 x 64

        nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.3, inplace=True),
        # Layer Output: 64 x 32 x 32

        nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.3, inplace=True),
        # Layer Output: 128 x 16 x 16

        nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(0.3, inplace=True),
        # Layer Output: 256 x 8 x 8

        nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(0.3, inplace=True),
        # Layer Output: 512 x 4 x 4

        # With a 4x4, we can condense the channels into a 1 x 1 x 1 to produce output
        nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
        nn.Flatten(),
        nn.Sigmoid()
    )


# Same as Discriminator 1, but with smaller kernel size
def disc_3():
    return nn.Sequential(
        # Input is 3 x 256 x 256
        nn.Conv2d(3, 16, kernel_size=2, stride=2, padding=0, bias=False),
        nn.BatchNorm2d(16),
        nn.LeakyReLU(0.3, inplace=True),
        # Layer Output: 16 x 128 x 128

        nn.Conv2d(16, 32, kernel_size=2, stride=2, padding=0, bias=False),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(0.3, inplace=True),
        # Layer Output: 32 x 64 x 64

        nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=0, bias=False),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.3, inplace=True),
        # Layer Output: 64 x 32 x 32

        nn.Conv2d(64, 128, kernel_size=2, stride=2, padding=0, bias=False),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.3, inplace=True),
        # Layer Output: 128 x 16 x 16

        nn.Conv2d(128, 256, kernel_size=2, stride=2, padding=0, bias=False),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(0.3, inplace=True),
        # Layer Output: 256 x 8 x 8

        nn.Conv2d(256, 512, kernel_size=2, stride=2, padding=0, bias=False),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(0.3, inplace=True),
        # Layer Output: 512 x 4 x 4

        # Additional layer to make it 2x2
        nn.Conv2d(512, 1024, kernel_size=2, stride=2, padding=0, bias=False),
        nn.BatchNorm2d(1024),
        nn.LeakyReLU(0.3, inplace=True),
        # Layer Output: 512 x 2 x 2

        # With a 2x2, we can condense the channels into a 1 x 1 x 1 to produce output
        nn.Conv2d(1024, 1, kernel_size=2, stride=1, padding=0, bias=False),
        nn.Flatten(),
        nn.Sigmoid()
    )


def disc_5():
    return nn.Sequential(
        # Input is 3 x 256 x 256
        nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(0.15, inplace=True),
        # Layer Output: 64 x 128 x 128

        nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.15, inplace=True),
        # Layer Output: 128 x 64 x 64

        nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.15, inplace=True),
        # Layer Output: 256 x 32 x 32

        nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(0.15, inplace=True),
        # Layer Output: 256 x 16 x 16

        nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(0.15, inplace=True),
        # Layer Output: 256 x 8 x 8

        nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(0.15, inplace=True),
        # Layer Output: 256 x 4 x 4

        # With a 4x4, we can condense the channels into a 1 x 1 x 1 to produce output
        nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0, bias=False),
        nn.Flatten(),
        nn.Sigmoid()
    )