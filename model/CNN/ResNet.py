import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, GlobalAveragePooling2D, Dense, Add

class ResNet2LayerBlock(tf.keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu"):
        super(ResNet2LayerBlock, self).__init__()
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

class ResNet3LayerBlock(tf.keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu"):
        super(ResNet3LayerBlock, self).__init__()
        self.conv1 = Conv2D(filters, kernel_size=(1, 1), strides=strides, padding="same")
        self.bn1 = BatchNormalization()
        self.activation1 = Activation(activation)
        self.conv2 = Conv2D(filters, kernel_size=(3, 3), padding="same")
        self.bn2 = BatchNormalization()
        self.activation2 = Activation(activation)
        self.conv3 = Conv2D(filters * 4, kernel_size=(1, 1))
        self.bn3 = BatchNormalization()

        self.shortcut = tf.keras.Sequential()
        if strides != 1:
            self.shortcut.add(Conv2D(filters * 4, kernel_size=(1, 1), strides=strides))
            self.shortcut.add(BatchNormalization())

        self.activation3 = Activation(activation)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation2(x)
        x = self.conv3(x)
        x = self.bn3(x)

        shortcut = self.shortcut(inputs)
        x = Add()([x, shortcut])
        x = self.activation3(x)
        return x
    

class ResNet18(tf.keras.Model):
    def __init__(self, input_shape, num_classes):
        super(ResNet18, self).__init__()
        self.conv1 = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding="same", input_shape=input_shape)
        self.bn1 = BatchNormalization()
        self.activation1 = Activation("relu")
        self.pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")
        
        self.block1 = ResNet2LayerBlock(64)
        self.block2 = ResNet2LayerBlock(64)
        self.block3 = ResNet2LayerBlock(128, strides=2)
        self.block4 = ResNet2LayerBlock(128)
        self.block5 = ResNet2LayerBlock(256, strides=2)
        self.block6 = ResNet2LayerBlock(256)
        self.block7 = ResNet2LayerBlock(512, strides=2)
        self.block8 = ResNet2LayerBlock(512)

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

        # Using ResNetBlock2Layer for ResNet-34
        self.block1 = ResNet2LayerBlock(64, strides=1)
        self.block2 = ResNet2LayerBlock(64)
        self.block3 = ResNet2LayerBlock(64)

        self.block4 = ResNet2LayerBlock(128, strides=2)
        self.block5 = ResNet2LayerBlock(128)
        self.block6 = ResNet2LayerBlock(128)
        self.block7 = ResNet2LayerBlock(128)

        self.block8 = ResNet2LayerBlock(256, strides=2)
        self.block9 = ResNet2LayerBlock(256)
        self.block10 = ResNet2LayerBlock(256)
        self.block11 = ResNet2LayerBlock(256)
        self.block12 = ResNet2LayerBlock(256)
        self.block13 = ResNet2LayerBlock(256)

        self.block14 = ResNet2LayerBlock(512, strides=2)
        self.block15 = ResNet2LayerBlock(512)
        self.block16 = ResNet2LayerBlock(512)


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

class ResNet50(tf.keras.Modconv1x1_1el):
    def __init__(self, input_shape, num_classes):
        super(ResNet50, self).__init__()
        self.conv1 = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding="same", input_shape=input_shape)
        self.bn1 = BatchNormalization()
        self.activation1 = Activation("relu")
        self.pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")

        self.block1 = ResNet3LayerBlock(64, strides=1)
        self.block2 = ResNet3LayerBlock(64)
        self.block3 = ResNet3LayerBlock(128, strides=2)
        self.block4 = ResNet3LayerBlock(128)
        self.block5 = ResNet3LayerBlock(256, strides=2)
        self.block6 = ResNet3LayerBlock(256)
        self.block7 = ResNet3LayerBlock(512, strides=2)
        self.block8 = ResNet3LayerBlock(512)

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
    
class ResNet101(tf.keras.Model):
    def __init__(self, input_shape, num_classes):
        super(ResNet101, self).__init__()
        self.conv1 = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding="same", input_shape=input_shape)
        self.bn1 = BatchNormalization()
        self.activation1 = Activation("relu")
        self.pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")

        self.block1 = ResNet3LayerBlock(64, strides=1)
        self.block2 = ResNet3LayerBlock(64)
        self.block3 = ResNet3LayerBlock(64)

        self.block4 = ResNet3LayerBlock(128, strides=2)
        self.block5 = ResNet3LayerBlock(128)
        self.block6 = ResNet3LayerBlock(128)
        self.block7 = ResNet3LayerBlock(128)

        self.block8 = ResNet3LayerBlock(256, strides=2)
        self.block9 = ResNet3LayerBlock(256)
        self.block10 = ResNet3LayerBlock(256)
        self.block11 = ResNet3LayerBlock(256)
        self.block12 = ResNet3LayerBlock(256)
        self.block13 = ResNet3LayerBlock(256)
        self.block14 = ResNet3LayerBlock(256)

        self.block15 = ResNet3LayerBlock(512, strides=2)
        self.block16 = ResNet3LayerBlock(512)
        self.block17 = ResNet3LayerBlock(512)

        self.block18 = ResNet3LayerBlock(1024, strides=2)
        self.block19 = ResNet3LayerBlock(1024)
        self.block20 = ResNet3LayerBlock(1024)
        self.block21 = ResNet3LayerBlock(1024)
        self.block22 = ResNet3LayerBlock(1024)
        self.block23 = ResNet3LayerBlock(1024)
        self.block24 = ResNet3LayerBlock(1024)
        self.block25 = ResNet3LayerBlock(1024)
        self.block26 = ResNet3LayerBlock(1024)
        self.block27 = ResNet3LayerBlock(1024)
        self.block28 = ResNet3LayerBlock(1024)

        self.block29 = ResNet3LayerBlock(2048, strides=2)
        self.block30 = ResNet3LayerBlock(2048)
        self.block31 = ResNet3LayerBlock(2048)

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
        x = self.block17(x)

        x = self.block18(x)
        x = self.block19(x)
        x = self.block20(x)
        x = self.block21(x)
        x = self.block22(x)
        x = self.block23(x)
        x = self.block24(x)
        x = self.block25(x)
        x = self.block26(x)
        x = self.block27(x)
        x = self.block28(x)

        x = self.block29(x)
        x = self.block30(x)
        x = self.block31(x)

        x = self.global_avg_pool(x)
        x = self.dense(x)
        return x

class ResNet152(tf.keras.Model):
    def __init__(self, input_shape, num_classes):
        super(ResNet152, self).__init__()
        self.conv1 = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding="same", input_shape=input_shape)
        self.bn1 = BatchNormalization()
        self.activation1 = Activation("relu")
        self.pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")

        self.block1 = ResNet3LayerBlock(64, strides=1)
        self.block2 = ResNet3LayerBlock(64)
        self.block3 = ResNet3LayerBlock(64)

        self.block4 = ResNet3LayerBlock(128, strides=2)
        self.block5 = ResNet3LayerBlock(128)
        self.block6 = ResNet3LayerBlock(128)
        self.block7 = ResNet3LayerBlock(128)

        self.block8 = ResNet3LayerBlock(256, strides=2)
        self.block9 = ResNet3LayerBlock(256)
        self.block10 = ResNet3LayerBlock(256)
        self.block11 = ResNet3LayerBlock(256)
        self.block12 = ResNet3LayerBlock(256)
        self.block13 = ResNet3LayerBlock(256)
        self.block14 = ResNet3LayerBlock(256)

        self.block15 = ResNet3LayerBlock(512, strides=2)
        self.block16 = ResNet3LayerBlock(512)
        self.block17 = ResNet3LayerBlock(512)

        self.block18 = ResNet3LayerBlock(1024, strides=2)
        self.block19 = ResNet3LayerBlock(1024)
        self.block20 = ResNet3LayerBlock(1024)
        self.block21 = ResNet3LayerBlock(1024)
        self.block22 = ResNet3LayerBlock(1024)
        self.block23 = ResNet3LayerBlock(1024)
        self.block24 = ResNet3LayerBlock(1024)
        self.block25 = ResNet3LayerBlock(1024)
        self.block26 = ResNet3LayerBlock(1024)
        self.block27 = ResNet3LayerBlock(1024)
        self.block28 = ResNet3LayerBlock(1024)

        self.block29 = ResNet3LayerBlock(2048, strides=2)
        self.block30 = ResNet3LayerBlock(2048)
        self.block31 = ResNet3LayerBlock(2048)

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
        x = self.block17(x)

        x = self.block18(x)
        x = self.block19(x)
        x = self.block20(x)
        x = self.block21(x)
        x = self.block22(x)
        x = self.block23(x)
        x = self.block24(x)
        x = self.block25(x)
        x = self.block26(x)
        x = self.block27(x)
        x = self.block28(x)

        x = self.block29(x)
        x = self.block30(x)
        x = self.block31(x)

        x = self.global_avg_pool(x)
        x = self.dense(x)
        return x

