from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Reshape, Conv2DTranspose, LeakyReLU, BatchNormalization,Conv2D,Flatten,Dropout

class Generator(Model):
    def __init__(self, noise_shape):
        super(Generator, self).__init__()

        self.noise_shape = noise_shape

        self.dense = Dense(4*4*512, input_shape=[self.noise_shape])
        self.reshape = Reshape([4, 4, 512])
        self.conv1 = Conv2DTranspose(256, kernel_size=4, strides=2, padding="same")
        self.leaky_relu1 = LeakyReLU(alpha=0.2)
        self.batch_norm1 = BatchNormalization()
        self.conv2 = Conv2DTranspose(128, kernel_size=4, strides=2, padding="same")
        self.leaky_relu2 = LeakyReLU(alpha=0.2)
        self.batch_norm2 = BatchNormalization()
        self.conv3 = Conv2DTranspose(64, kernel_size=4, strides=2, padding="same")
        self.leaky_relu3 = LeakyReLU(alpha=0.2)
        self.batch_norm3 = BatchNormalization()
        self.conv4 = Conv2DTranspose(3, kernel_size=4, strides=2, padding="same", activation='sigmoid')

    def call(self, inputs):
        x = self.dense(inputs)
        x = self.reshape(x)
        x = self.conv1(x)
        x = self.leaky_relu1(x)
        x = self.batch_norm1(x)
        x = self.conv2(x)
        x = self.leaky_relu2(x)
        x = self.batch_norm2(x)
        x = self.conv3(x)
        x = self.leaky_relu3(x)
        x = self.batch_norm3(x)
        outputs = self.conv4(x)
        return outputs
class Discriminator(Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = Conv2D(32, kernel_size=4, strides=2, padding="same", input_shape=[64, 64, 3])
        self.conv2 = Conv2D(64, kernel_size=4, strides=2, padding="same")
        self.leaky_relu1 = LeakyReLU(0.2)
        self.batch_norm1 = BatchNormalization()
        self.conv3 = Conv2D(128, kernel_size=4, strides=2, padding="same")
        self.leaky_relu2 = LeakyReLU(0.2)
        self.batch_norm2 = BatchNormalization()
        self.conv4 = Conv2D(256, kernel_size=4, strides=2, padding="same")
        self.leaky_relu3 = LeakyReLU(0.2)
        self.flatten = Flatten()
        self.dropout = Dropout(0.5)
        self.dense = Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.leaky_relu1(x)
        x = self.batch_norm1(x)
        x = self.conv3(x)
        x = self.leaky_relu2(x)
        x = self.batch_norm2(x)
        x = self.conv4(x)
        x = self.leaky_relu3(x)
        x = self.flatten(x)
        x = self.dropout(x)
        outputs = self.dense(x)
        return outputs
