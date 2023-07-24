from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

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
