import tensorflow as tf
from tensorflow.keras.layers import  Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate,Layer, BatchNormalization
from tensorflow.keras.layers.core import Activation

#https://medium.com/geekculture/u-net-implementation-from-scratch-using-tensorflow-b4342266e406

class EncoderBlock(Layer):
    def __init__(self, num_filters, dropout_prob=0.3, max_pooling=True):
        super(EncoderBlock, self).__init__()
        self.conv1 = Conv2D(num_filters, 3, padding='same')
        self.relu1 = Activation('relu')
        self.conv2 = Conv2D(num_filters, 3, padding='same')
        self.relu2 = Activation('relu')
        self.bn1 = BatchNormalization(training=False)
        self.pool = MaxPooling2D(pool_size=(2, 2), strides=2)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool(x)
        return x

class DecoderBlock(Layer):
    def __init__(self, num_filters):
        super(DecoderBlock, self).__init__()
        self.upsample = Conv2DTranspose(num_filters, (2, 2), strides=2, padding='same')
        self.concat = Concatenate()
        self.conv1 = Conv2D(num_filters, 3, padding='same')
        self.relu1 = Activation('relu')
        self.conv2 = Conv2D(num_filters, 3, padding='same')
        self.relu2 = Activation('relu')

    def call(self, inputs, skip_features):
        x = self.upsample(inputs)
        skip_features = tf.image.resize(skip_features, size=(x.shape[1], x.shape[2]))
        x = self.concat([x, skip_features])
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x


class UNet(tf.keras.Model):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.num_classes = num_classes
        
        # Contracting Path
        self.s1 = EncoderBlock(64)
        self.s2 = EncoderBlock(128)
        self.s3 = EncoderBlock(256)
        self.s4 = EncoderBlock(512)
        

        # Bottom of the U
        self.b1 = Conv2D(1024, (3, 3), padding="same", activation="relu")
        self.b2 = Conv2D(1024, (3, 3), padding="same", activation="relu")

        # Expansive Path
        self.s5 = DecoderBlock(512)
        self.s6 = DecoderBlock(256)
        self.s7 = DecoderBlock(128)
        self.s8 = DecoderBlock(64)
    
        # Output Layer
        self.output_layer = Conv2D(self.num_classes, (1, 1), activation="softmax")

    def call(self, inputs):
        s1 = self.s1(inputs)
        s2 = self.s2(s1)
        s3 = self.s3(s2)
        s4 = self.s4(s3)
        
        b1 = self.b1(s4)
        b2 = self.b2(b1)
        
        s5 = DecoderBlock(b1, s4)
        s6 = DecoderBlock(s5, s3)
        s7 = DecoderBlock(s6, s2)
        s8 = DecoderBlock(s7, s1)
        
        outputs = self.output_layer(s8)

        return outputs