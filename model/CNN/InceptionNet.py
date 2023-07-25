import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras import Model
from tensorflow.keras.layers.core import Activation
from tensorflow.keras.regularizers import l2

class ConvModule(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, stride, chanDim, padding="same", reg=0.0005, name=None):
        super(ConvModule, self).__init__(name=name)
        self.conv = Conv2D(filters, kernel_size, strides=stride, padding=padding, kernel_regularizer=l2(reg))
        self.bn = BatchNormalization(axis=chanDim)
        self.relu = Activation("relu")

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class InceptionModule(tf.keras.layers.Layer):
    def __init__(self, num1x1, num3x3Reduce, num3x3, num5x5Reduce, num5x5, num1x1Proj, chanDim, reg=0.0005, name=None):
        super(InceptionModule, self).__init__(name=name)

        # Branch 1: 1x1 convolution for reducing dimensionality before the main convolution
        self.first = ConvModule(num1x1, (1, 1), 1, chanDim, reg=reg, name=name + "_first")

        # Branch 2: 1x1 convolution (dimension reduction) followed by 3x3 convolution
        self.second1 = ConvModule(num3x3Reduce, (1, 1), 1, chanDim, reg=reg, name=name + "_second1")
        self.second2 = ConvModule(num3x3, (3, 3), 1, chanDim, reg=reg, name=name + "_second2")

        # Branch 3: 1x1 convolution (dimension reduction) followed by 5x5 convolution
        self.third1 = ConvModule(num5x5Reduce, (1, 1), 1, chanDim, reg=reg, name=name + "_third1")
        self.third2 = ConvModule(num5x5, (5, 5), 1, chanDim, reg=reg, name=name + "_third2")

        # Branch 4: MaxPooling followed by 1x1 convolution for dimension reduction
        self.fourth_pool = MaxPooling2D((3, 3), strides=(1, 1), padding="same", name=name + "_pool")
        self.fourth = ConvModule(num1x1Proj, (1, 1), 1, chanDim, reg=reg, name=name + "_fourth")

    def call(self, x):
        # Pass input tensor through each branch
        first_out = self.first(x)
        second_out = self.second1(x)
        second_out = self.second2(second_out)
        third_out = self.third1(x)
        third_out = self.third2(third_out)
        fourth_out = self.fourth_pool(x)
        fourth_out = self.fourth(fourth_out)

        # Concatenate the outputs along the channel dimension
        x = Concatenate([first_out, second_out, third_out, fourth_out], axis=self.fourth.filters, name=self.name + "_mixed")
        return x

class InceptionV1(tf.keras.Model):
    def __init__(self, num_classes, reg=0.0005):
        super(InceptionV1, self).__init__()

        # Define the input layer
        self.input_layer = Input(shape=(224, 224, 3))

        # Stage 1: Initial Convolution and Pooling
        x = ConvModule(64, (7, 7), 2, chanDim=-1, padding="valid", reg=reg, name="stage1_conv")(self.input_layer)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding="valid", name="stage1_pool")(x)

        # Stage 2: Inception Module followed by MaxPooling
        x = InceptionModule(64, 96, 128, 16, 32, 32, -1, reg=reg, name="stage2_inception")(x)
        x = InceptionModule(128, 128, 192, 32, 96, 64, -1, reg=reg, name="stage2_inception_2")(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding="valid", name="stage2_pool")(x)

        # Stage 3: Multiple Inception Modules
        x = InceptionModule(192, 96, 208, 16, 48, 64, -1, reg=reg, name="stage3_inception")(x)
        x = InceptionModule(160, 112, 224, 24, 64, 64, -1, reg=reg, name="stage3_inception_2")(x)
        x = InceptionModule(128, 128, 256, 24, 64, 64, -1, reg=reg, name="stage3_inception_3")(x)
        x = InceptionModule(112, 144, 288, 32, 64, 64, -1, reg=reg, name="stage3_inception_4")(x)
        x = InceptionModule(256, 160, 320, 32, 128, 128, -1, reg=reg, name="stage3_inception_5")(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding="valid", name="stage3_pool")(x)

        # Stage 4: Multiple Inception Modules
        x = InceptionModule(256, 160, 320, 32, 128, 128, -1, reg=reg, name="stage4_inception")(x)
        x = InceptionModule(384, 192, 384, 48, 128, 128, -1, reg=reg, name="stage4_inception_2")(x)

        # Stage 5: Average Pooling, Dropout, and Fully Connected Layers
        x = AveragePooling2D((7, 7), strides=1, name="avg_pool")(x)
        x = Flatten(name="flatten")(x)
        x = Dropout(0.4, name="dropout")(x)
        x = Dense(num_classes, activation="softmax", kernel_regularizer=tf.keras.regularizers.l2(reg), name="predictions")(x)

        # Create the model
        self.model = tf.keras.Model(inputs=self.input_layer, outputs=x)

    def call(self, inputs):
        return self.model(inputs)
