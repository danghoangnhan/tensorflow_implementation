from torch.nn import ConvTranspose2d, Tanh, Module, Sequential, ConvTranspose2d, BatchNorm2d, ReLU, Tanh

from SupervisedLearning.DCGan import Deconv

seed_size = 16


def gen_1():
    return Sequential(
        # Input seed_size x 1 x 1
        ConvTranspose2d(seed_size, 512, kernel_size=4, padding=0, stride=1, bias=False),
        BatchNorm2d(512),
        ReLU(True),
        # Layer output: 512 x 4 x 4

        ConvTranspose2d(512, 256, kernel_size=4, padding=1, stride=2, bias=False),
        BatchNorm2d(256),
        ReLU(True),
        # Layer output: 256 x 8 x 8

        ConvTranspose2d(256, 128, kernel_size=4, padding=1, stride=2, bias=False),
        BatchNorm2d(128),
        ReLU(True),
        # Layer output: 128 x 16 x 16

        ConvTranspose2d(128, 64, kernel_size=4, padding=1, stride=2, bias=False),
        BatchNorm2d(64),
        ReLU(True),
        # Layer output: 64 x 32 x 32

        ConvTranspose2d(64, 32, kernel_size=4, padding=1, stride=2, bias=False),
        BatchNorm2d(32),
        ReLU(True),
        # Layer output: 32 x 64 x 64

        ConvTranspose2d(32, 16, kernel_size=4, padding=1, stride=2, bias=False),
        BatchNorm2d(16),
        ReLU(True),
        # Layer output: 16 x 128 x 128

        ConvTranspose2d(16, 3, kernel_size=4, padding=1, stride=2, bias=False),
        Tanh()
        # Output: 3 x 256 x 256
    )


# Generator matching Discriminator 3
def gen_3():
    return Sequential(
        # Input seed_size x 1 x 1
        ConvTranspose2d(seed_size, 1024, kernel_size=2, padding=0, stride=1, bias=False),
        BatchNorm2d(1024),
        ReLU(True),
        # Layer output: 1024 x 2 x 2

        ConvTranspose2d(1024, 512, kernel_size=2, padding=0, stride=2, bias=False),
        BatchNorm2d(512),
        ReLU(True),
        # Layer output: 512 x 4 x 4

        ConvTranspose2d(512, 256, kernel_size=2, padding=0, stride=2, bias=False),
        BatchNorm2d(256),
        ReLU(True),
        # Layer output: 256 x 8 x 8

        ConvTranspose2d(256, 128, kernel_size=2, padding=0, stride=2, bias=False),
        BatchNorm2d(128),
        ReLU(True),
        # Layer output: 128 x 16 x 16

        ConvTranspose2d(128, 64, kernel_size=2, padding=0, stride=2, bias=False),
        BatchNorm2d(64),
        ReLU(True),
        # Layer output: 64 x 32 x 32

        ConvTranspose2d(64, 32, kernel_size=2, padding=0, stride=2, bias=False),
        BatchNorm2d(32),
        ReLU(True),
        # Layer output: 32 x 64 x 64

        ConvTranspose2d(32, 16, kernel_size=2, padding=0, stride=2, bias=False),
        BatchNorm2d(16),
        ReLU(True),
        # Layer output: 16 x 128 x 128

        ConvTranspose2d(16, 3, kernel_size=2, padding=0, stride=2, bias=False),
        Tanh()
        # Output: 3 x 256 x 256
    )


# Generator with lots of upsampling weirdness
def gen_4():
    return Sequential(
        # Input seed_size x 1 x 1
        ConvTranspose2d(seed_size, 1024, kernel_size=2, padding=0, stride=1, bias=False),
        BatchNorm2d(1024),
        ReLU(True),
        # Layer output: 1024 x 2 x 2

        ConvTranspose2d(1024, 512, kernel_size=2, padding=0, stride=2, bias=False),
        BatchNorm2d(512),
        ReLU(True),
        # Layer output: 512 x 4 x 4

        ConvTranspose2d(512, 256, kernel_size=2, padding=0, stride=2, bias=False),
        BatchNorm2d(256),
        ReLU(True),
        # Layer output: 256 x 8 x 8

        ConvTranspose2d(256, 128, kernel_size=3, padding=1, stride=9, bias=False),
        BatchNorm2d(128),
        ReLU(True),
        # Layer output: 128 x 64 x 64

        ConvTranspose2d(128, 3, kernel_size=4, padding=0, stride=4, bias=False),
        Tanh()
        # Output: 3 x 256 x 256
    )


def gen_5():
    return Sequential(
        # Input seed_size x 1 x 1
        ConvTranspose2d(seed_size, 256, kernel_size=4, padding=0, stride=1, bias=False),
        BatchNorm2d(256),
        ReLU(True),
        # Layer output: 256 x 4 x 4

        ConvTranspose2d(256, 256, kernel_size=4, padding=1, stride=2, bias=False),
        BatchNorm2d(256),
        ReLU(True),
        # Layer output: 256 x 8 x 8

        ConvTranspose2d(256, 256, kernel_size=4, padding=1, stride=2, bias=False),
        BatchNorm2d(256),
        ReLU(True),
        # Layer output: 256 x 16 x 16

        ConvTranspose2d(256, 128, kernel_size=4, padding=1, stride=2, bias=False),
        BatchNorm2d(128),
        ReLU(True),
        # Layer output: 128 x 32 x 32

        ConvTranspose2d(128, 64, kernel_size=4, padding=1, stride=2, bias=False),
        BatchNorm2d(64),
        ReLU(True),
        # Layer output: 128 x 64 x 64

        ConvTranspose2d(64, 32, kernel_size=4, padding=1, stride=2, bias=False),
        BatchNorm2d(32),
        ReLU(True),
        # Layer output: 64 x 128 x 128

        ConvTranspose2d(32, 3, kernel_size=4, padding=1, stride=2, bias=False),
        Tanh()
        # Output: 3 x 256 x 256
    )


def gen_64_1():
    return Sequential(
        # Input seed_size x 1 x 1
        ConvTranspose2d(seed_size, 256, kernel_size=4, padding=0, stride=1, bias=False),
        BatchNorm2d(256),
        ReLU(True),
        # Layer output: 256 x 4 x 4

        ConvTranspose2d(256, 128, kernel_size=4, padding=1, stride=2, bias=False),
        BatchNorm2d(128),
        ReLU(True),
        # Layer output: 128 x 8 x 8

        ConvTranspose2d(128, 64, kernel_size=4, padding=1, stride=2, bias=False),
        BatchNorm2d(64),
        ReLU(True),
        # Layer output: 64 x 16 x 16

        ConvTranspose2d(64, 32, kernel_size=4, padding=1, stride=2, bias=False),
        BatchNorm2d(32),
        ReLU(True),
        # Layer output: 32 x 32 x 32

        ConvTranspose2d(32, 3, kernel_size=4, padding=1, stride=2, bias=False),
        Tanh()
        # Output: 3 x 64 x 64
    )


gen_64_2 = Sequential(
    # Input seed_size x 1 x 1
    ConvTranspose2d(seed_size, 128, kernel_size=4, padding=0, stride=1, bias=False),
    BatchNorm2d(128),
    ReLU(True),
    # Layer output: 256 x 4 x 4

    ConvTranspose2d(128, 128, kernel_size=4, padding=1, stride=2, bias=False),
    BatchNorm2d(128),
    ReLU(True),
    # Layer output: 128 x 8 x 8

    ConvTranspose2d(128, 128, kernel_size=4, padding=1, stride=2, bias=False),
    BatchNorm2d(128),
    ReLU(True),
    # Layer output: 64 x 16 x 16

    ConvTranspose2d(128, 64, kernel_size=4, padding=1, stride=2, bias=False),
    BatchNorm2d(64),
    ReLU(True),
    # Layer output: 32 x 32 x 32

    ConvTranspose2d(64, 3, kernel_size=4, padding=1, stride=2, bias=False),
    Tanh()
    # Output: 3 x 64 x 64
)


class Generator(Module):
    def __init__(self, z=100, nc=64):
        super(Generator, self).__init__()
        self.net = Sequential(
            Deconv(z, nc * 8, 4, 1, 0),
            Deconv(nc * 8, nc * 4, 4, 2, 1),
            Deconv(nc * 4, nc * 2, 4, 2, 1),
            Deconv(nc * 2, nc, 4, 2, 1),
            ConvTranspose2d(nc, 3, 4, 2, 1, bias=False),
            Tanh()
        )

    def forward(self, input):
        return self.net(input)
