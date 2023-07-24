import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, Dropout, Flatten, Dense

class InceptionModule(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(InceptionModule, self).__init__()
        
        self.conv1x1_1 = Conv2D(filters[0], (1, 1), padding='same', activation='relu')
        self.conv1x1_2 = Conv2D(filters[1], (1, 1), padding='same', activation='relu')
        self.conv1x1_3 = Conv2D(filters[3], (1, 1), padding='same', activation='relu')
        self.conv1x1_4 = Conv2D(filters[5], (1, 1), padding='same', activation='relu')
    
        self.conv3x3 = Conv2D(filters[2], (3, 3), padding='same', activation='relu')
        self.conv5x5 = Conv2D(filters[4], (5, 5), padding='same', activation='relu')
        self.max_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")

    def call(self, x):
        conv1x1_1 = self.conv1x1_1(x)
        conv1x1_2 = self.conv1x1_2(x)
        conv3x3 = self.conv3x3(conv1x1_2)
        conv1x1_3 = self.conv1x1_3(x)
        conv5x5 = self.conv5x5(conv1x1_3)
        maxpool = self.max_pool(x)
        conv1x1_4 = self.conv1x1_4(maxpool)

        inception_block = Concatenate(axis=-1)([conv1x1_1, conv3x3, conv5x5, conv1x1_4])
        return inception_block

class InceptionV1(tf.keras.Model):
    def __init__(self, input_shape=(224, 224, 3), num_classes=1000):
        super(InceptionV1, self).__init__()
        self.conv1 = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')
        self.pool1 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')
        self.conv2 = Conv2D(192, (3, 3), padding='same', activation='relu')
        self.pool2 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')

        self.inception3a = InceptionModule([64, 128, 128, 64, 64, 32])
        self.inception3b = InceptionModule([128, 192, 96, 128, 128, 64])

        self.inception4a = InceptionModule([192, 208, 48, 96, 96, 64])
        self.inception4b = InceptionModule([160, 224, 64, 112, 112, 64])
        self.inception4c = InceptionModule([128, 256, 64, 128, 128, 64])
        self.inception4d = InceptionModule([112, 288, 64, 144, 144, 64])
        self.inception4e = InceptionModule([256, 320, 128, 160, 160, 128])

        self.inception5a = InceptionModule([256, 320, 128, 160, 160, 128])
        self.inception5b = InceptionModule([384, 384, 128, 192, 192, 128])

        self.avg_pool = AveragePooling2D((7, 7))
        self.dropout = Dropout(0.4)
        self.dense = Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avg_pool(x)
        x = self.dropout(x)
        x = self.dense(Flatten()(x))
        return x
    
    
class InceptionBlock(tf.keras.layers.Layer):
    def __init__(self, conv1_filters, conv3_filters, conv5_filters):
            super(InceptionBlock, self).__init__()
            self.conv1x1_1 = Conv2D(conv1_filters[0], kernel_size=(1, 1), padding="same", activation="relu")
            self.conv1x1_2 = Conv2D(conv1_filters[1], kernel_size=(1, 1), padding="same", activation="relu")
            self.conv1x1_3 = Conv2D(conv1_filters[2], kernel_size=(1, 1), padding="same", activation="relu")
            self.conv1x1_4 = Conv2D(conv1_filters[3], kernel_size=(1, 1), padding="same", activation="relu")

            self.conv3x3 = Conv2D(conv3_filters[0], kernel_size=(1, 1), padding="same", activation="relu")
            self.conv3x3_1 = Conv2D(conv3_filters[1], kernel_size=(3, 3), padding="same", activation="relu")
            self.conv3x3_2 = Conv2D(conv3_filters[2], kernel_size=(3, 3), padding="same", activation="relu")
            self.conv3x3_3 = Conv2D(conv3_filters[3], kernel_size=(3, 3), padding="same", activation="relu")

            self.conv5x5 = Conv2D(conv5_filters[0], kernel_size=(1, 1), padding="same", activation="relu")
            self.conv5x5_1 = Conv2D(conv5_filters[1], kernel_size=(5, 5), padding="same", activation="relu")

            self.max_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")
            self.max_pool_conv = Conv2D(conv5_filters[2], kernel_size=(1, 1), padding="same", activation="relu")

    def call(self, x):
            branch1 = self.conv1x1_1(x)
            branch1 = self.conv1x1_2(branch1)
            branch2 = self.conv3x3_1(self.conv3x3(x))
            branch3 = self.conv5x5_1(self.conv5x5(x))
            branch4 = self.max_pool_conv(self.max_pool(x))
            return concatenate([branch1, branch2, branch3, branch4])

class InceptionV2(tf.keras.Model):
    def __init__(self, num_classes):
        super(InceptionV2, self).__init__()

        # Stem block
        self.stem_block = tf.keras.Sequential([
            Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding="same", activation="relu"),
            MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")
        ])

        # Inception blocks
        self.inception_block1 = InceptionBlock([64, 64, 64, 64], [64, 96, 96, 32, 32], [128, 128, 128])
        self.inception_block2 = InceptionBlock([64, 64, 96, 64], [64, 96, 96, 64, 64], [192, 128, 128])
        self.inception_block3 = InceptionBlock([128, 128, 128, 128], [128, 160, 160, 96, 96], [256, 128, 128])
        self.inception_block4 = InceptionBlock([128, 128, 128, 128], [128, 160, 160, 128, 128], [384, 128, 128])

        # Global Average Pooling and Classifier
        self.global_avg_pool = GlobalAveragePooling2D()
        self.classifier = Dense(num_classes, activation="softmax")

    def call(self, inputs):
        x = self.stem_block(inputs)
        x = self.inception_block1(x)
        x = self.inception_block2(x)
        x = self.inception_block3(x)
        x = self.inception_block4(x)
        x = self.global_avg_pool(x)
        x = self.classifier(x)
        return x
