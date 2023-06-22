from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, BatchNormalization, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K

class AlexNet(Model):
    def __init__(self, width, height, depth, classes, reg=0.0002):
        super(AlexNet, self).__init__()

        self.width = width
        self.height = height
        self.depth = depth
        self.classes = classes
        self.reg = reg

        # Set the input shape and channels dimension based on image data format
        input_shape = (self.height, self.width, self.depth)
        chan_dim = -1
        if K.image_data_format() == "channels_first":
            input_shape = (self.depth, self.height, self.width)
            chan_dim = 1

        # Block #1: first CONV => RELU => POOL layer set
        self.conv1 = Conv2D(96, (11, 11), strides=(4, 4),input_shape=input_shape, padding="same",kernel_regularizer=regularizers.l2(self.reg),activation="relu")
        self.bn1 = BatchNormalization(axis=chan_dim)
        self.pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))
        self.dropout1 = Dropout(0.25)

        # Block #2: second CONV => RELU => POOL layer set
        self.conv2 = Conv2D(256, (5, 5), padding="same", kernel_regularizer=regularizers.l2(self.reg),activation="relu")
        self.bn2 = BatchNormalization(axis=chan_dim)
        self.pool2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))
        self.dropout2 = Dropout(0.25)

        # Block #3: CONV => RELU => CONV => RELU => CONV => RELU
        self.conv3 = Conv2D(384, (3, 3), padding="same", kernel_regularizer=regularizers.l2(self.reg),activation="relu")
        self.bn3 = BatchNormalization(axis=chan_dim)
        self.conv4 = Conv2D(384, (3, 3), padding="same", kernel_regularizer=regularizers.l2(self.reg),activation="relu")
        self.bn4 = BatchNormalization(axis=chan_dim)
        self.conv5 = Conv2D(256, (3, 3), padding="same", kernel_regularizer=regularizers.l2(self.reg),activation="relu")
        self.bn5 = BatchNormalization(axis=chan_dim)
        self.pool3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))
        self.dropout3 = Dropout(0.25)

        # Block #4: first set of FC => RELU layers
        self.flatten = Flatten()
        self.fc1 = Dense(4096, kernel_regularizer=regularizers.l2(self.reg),activation="relu")
        self.bn6 = BatchNormalization()
        self.dropout4 = Dropout(0.5)

        # Block #5: second set of FC => RELU layers
        self.fc2 = Dense(4096, kernel_regularizer=regularizers.l2(self.reg),activation="relu")
        self.bn7 = BatchNormalization()
        self.dropout5 = Dropout(0.5)
        # Softmax classifier
        self.fc3 = Dense(self.classes, kernel_regularizer=regularizers.l2(self.reg),activation="softmax")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn6(x)
        x = self.dropout4(x)

        x = self.fc2(x)
        x = self.bn7(x)
        x = self.dropout5(x)

        x = self.fc3(x)
        return x
