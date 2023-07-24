from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, BatchNormalization, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential

class AlexNet(Model):
    def __init__(self, width, height, depth, classes, reg=0.0002):
        super(AlexNet, self).__init__()

        input_shape = (height, width, depth)
        chan_dim = -1
        if K.image_data_format() == "channels_first":
            input_shape = (depth, height, width)
            chan_dim = 1

        # Block #1: first CONV => RELU => POOL layer set
        self.add(Conv2D(96, (11, 11), strides=(4, 4), input_shape=input_shape, padding="same",
                        kernel_regularizer=regularizers.l2(reg), activation="relu"))
        self.add(BatchNormalization(axis=chan_dim))
        self.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        self.add(Dropout(0.25))
        
        # Block #2: second CONV => RELU => POOL layer set
        self.add(Conv2D(256, (5, 5), padding="same", kernel_regularizer=regularizers.l2(reg), activation="relu"))
        self.add(BatchNormalization(axis=chan_dim))
        self.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        self.add(Dropout(0.25))
        
        
        # Block #3: CONV => RELU => CONV => RELU => CONV => RELU
        self.add(Conv2D(384, (3, 3), padding="same", kernel_regularizer=regularizers.l2(reg), activation="relu"))
        self.add(BatchNormalization(axis=chan_dim))
        self.add(Conv2D(384, (3, 3), padding="same", kernel_regularizer=regularizers.l2(reg), activation="relu"))
        self.add(BatchNormalization(axis=chan_dim))
        self.add(Conv2D(256, (3, 3), padding="same", kernel_regularizer=regularizers.l2(reg), activation="relu"))
        self.add(BatchNormalization(axis=chan_dim))
        self.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        self.add(Dropout(0.25))

        # Block #4: first set of FC => RELU layers
        self.add(Flatten())
        self.add(Dense(4096, kernel_regularizer=regularizers.l2(reg), activation="relu"))
        self.add(BatchNormalization())
        self.add(Dropout(0.5))

        # Block #5: second set of FC => RELU layers
        self.add(Dense(4096, kernel_regularizer=regularizers.l2(reg), activation="relu"))
        self.add(BatchNormalization())
        self.add(Dropout(0.5))
        # Softmax classifier
        self.add(Dense(classes, kernel_regularizer=regularizers.l2(reg), activation="softmax"))

class ShallowNet(Sequential):
    def __init__(self, width, height, depth, classes):
        super(ShallowNet, self).__init__()

        input_shape = (height, width, depth)

        if K.image_data_format() == "channels_first":
            input_shape = (depth, height, width)

        self.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape))
        self.add(Activation("relu"))

        self.add(Flatten())
        self.add(Dense(classes))
        self.add(Activation("softmax"))