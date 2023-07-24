import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, GlobalAveragePooling2D, Dense, Add

class ResNetBlock(tf.keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu"):
        super(ResNetBlock, self).__init__()
        self.conv1 = Conv2D(filters, kernel_size=(3, 3), strides=strides, padding="same")
        self.bn1 = BatchNormalization()
        self.activation1 = Activation(activation)
        self.conv2 = Conv2D(filters, kernel_size=(3, 3), padding="same")
        self.bn2 = BatchNormalization()

        self.shortcut = tf.keras.Sequential()
        if strides != 1:
            self.shortcut.add(Conv2D(filters, kernel_size=(1, 1), strides=strides, padding="same"))
            self.shortcut.add(BatchNormalization())

        self.activation2 = Activation(activation)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.bn2(x)

        shortcut = self.shortcut(inputs)
        x = Add()([x, shortcut])
        x = self.activation2(x)
        return x

class ResNet18(tf.keras.Model):
    def __init__(self, input_shape, num_classes):
        super(ResNet18, self).__init__()
        self.conv1 = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding="same", input_shape=input_shape)
        self.bn1 = BatchNormalization()
        self.activation1 = Activation("relu")
        self.pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")

        self.block1 = ResNetBlock(64)
        self.block2 = ResNetBlock(64)
        self.block3 = ResNetBlock(128, strides=2)
        self.block4 = ResNetBlock(128)
        self.block5 = ResNetBlock(256, strides=2)
        self.block6 = ResNetBlock(256)
        self.block7 = ResNetBlock(512, strides=2)
        self.block8 = ResNetBlock(512)

        self.global_avg_pool = GlobalAveragePooling2D()
        self.dense = Dense(num_classes, activation="softmax")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.pool1(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)

        x = self.global_avg_pool(x)
        x = self.dense(x)
        return x

class ResNet34(tf.keras.Model):
    def __init__(self, input_shape, num_classes):
        super(ResNet34, self).__init__()
        self.conv1 = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding="same", input_shape=input_shape)
        self.bn1 = BatchNormalization()
        self.activation1 = Activation("relu")
        self.pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")

        self.block1 = ResNetBlock(64)
        self.block2 = ResNetBlock(64)
        self.block3 = ResNetBlock(64)

        self.block4 = ResNetBlock(128, strides=2)
        self.block5 = ResNetBlock(128)
        self.block6 = ResNetBlock(128)
        self.block7 = ResNetBlock(128)

        self.block8 = ResNetBlock(256, strides=2)
        self.block9 = ResNetBlock(256)
        self.block10 = ResNetBlock(256)
        self.block11 = ResNetBlock(256)
        self.block12 = ResNetBlock(256)
        self.block13 = ResNetBlock(256)

        self.block14 = ResNetBlock(512, strides=2)
        self.block15 = ResNetBlock(512)
        self.block16 = ResNetBlock(512)

        self.global_avg_pool = GlobalAveragePooling2D()
        self.dense = Dense(num_classes, activation="softmax")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.pool1(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)

        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)

        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)

        x = self.global_avg_pool(x)
        x = self.dense(x)
        return x
