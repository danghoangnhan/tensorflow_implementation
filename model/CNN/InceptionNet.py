import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D,Activation, MaxPooling2D, AveragePooling2D, Concatenate, Dropout, Flatten, Dense, BatchNormalization

class ConvModule(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding="same", activation='relu'):
        super(ConvModule, self).__init__()
        self.conv = Conv2D(filters, kernel_size, strides=strides, padding=padding)
        self.bn = BatchNormalization()
        self.activation = Activation(activation)

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class  DimensionReductionModule(tf.keras.layers.Layer):
    def __init__(self, num1x1,
                num3x3Reduce, num3x3,
                num5x5Reduce, num5x5,
                poolProj
        ):
        super(DimensionReductionModule, self).__init__()

        # Branch 1: 1x1 convolution for reducing dimensionality before the main convolution
        self.first = ConvModule(filters=num1x1,kernel_size=(1, 1),strides=1)
        # Branch 2: 1x1 convolution (dimension reduction) followed by 3x3 convolution
        self.branch_2 = tf.keras.Sequential([
            ConvModule(filters=num3x3Reduce,kernel_size=(1, 1),strides=1),
            ConvModule(filters=num3x3,kernel_size=(3, 3),strides=1)
        ]) 
        # Branch 3: 1x1 convolution (dimension reduction) followed by 5x5 convolution
        self.branch_3 = tf.keras.Sequential([
            ConvModule(filters=num5x5Reduce,kernel_size=(1, 1),strides=1),
            ConvModule(num5x5, (5, 5),1)
        ])
        # Branch 4: MaxPooling followed by 1x1 convolution for dimension reduction
        self.branch_4 = tf.keras.Sequential([
            MaxPooling2D((3, 3), strides=1, padding="same"),
            ConvModule(poolProj, (1, 1), 1)
        ])

    def call(self, x):
        first_out = self.first(x)
        second_out = self.branch_2(x)
        third_out = self.branch_3(x)
        fourth_out = self.branch_4(x)
        x = Concatenate(axis=-1)([first_out, second_out, third_out, fourth_out])
        return x

class InceptionV1(tf.keras.Model):
    def __init__(self, input_shape,num_classes):
        super(InceptionV1, self).__init__()
        self.model = tf.keras.Sequential([
            # Input(shape=input_shape),
            Conv2D(filters=64, kernel_size=(7,7), strides=2, padding="same",input_shape=input_shape),
            BatchNormalization(),
            Activation("relu"),
            # ConvModule(64, (7, 7),strides=2, padding="same",input_shape=input_shape),
            MaxPooling2D((3, 3), strides=2, padding="same"),
            DimensionReductionModule(num1x1=64,num3x3Reduce=96,num3x3=128,num5x5Reduce=16,num5x5=32,poolProj=32),
            DimensionReductionModule(num1x1=128,num3x3Reduce=128,num3x3=192,num5x5Reduce=32,num5x5=96,poolProj=64),
            MaxPooling2D((3, 3), strides=2, padding="same"),
            DimensionReductionModule(num1x1=192,num3x3Reduce=96,num3x3=208,num5x5Reduce=16,num5x5=48,poolProj=64),
            DimensionReductionModule(num1x1=160,num3x3Reduce=112,num3x3=224,num5x5Reduce=24,num5x5=64,poolProj=64),
            DimensionReductionModule(num1x1=128,num3x3Reduce=128,num3x3=256,num5x5Reduce=24,num5x5=64,poolProj=64),
            DimensionReductionModule(num1x1=112,num3x3Reduce=144,num3x3=288,num5x5Reduce=32,num5x5=64,poolProj=64),
            DimensionReductionModule(num1x1=256,num3x3Reduce=160,num3x3=320,num5x5Reduce=32,num5x5=128,poolProj=128),
            MaxPooling2D((3, 3), strides=2, padding="same"),            
            DimensionReductionModule(num1x1=256,num3x3Reduce=160,num3x3=320,num5x5Reduce=32,num5x5=128,poolProj=128),
            DimensionReductionModule(num1x1=384,num3x3Reduce=192,num3x3=384,num5x5Reduce=48,num5x5=128,poolProj=128),        
            AveragePooling2D((7, 7), strides=1),
            Flatten(),
            Dropout(0.4),
            Dense(num_classes, activation="softmax")
        ])
    def call(self, inputs):
        return self.model(inputs)