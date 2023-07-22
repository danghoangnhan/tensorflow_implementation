from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras import backend as K
from tensorflow.keras import Sequential
#source: https://www.kaggle.com/code/blurredmachine/vggnet-16-architecture-a-complete-guide?scriptVersionId=39674893
class VGG16(Sequential):
    def __init__(self,width, height, depth, classes=1000):
        super(VGG16, self).__init__()
        inputShape = (height, width, depth)

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            
        self.add(Conv2D(inputShape=inputShape,filters =64, kernel_size=(3, 3), padding="same", activation="relu"))
        self.add(Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"))
        self.add(MaxPooling2D((2, 2)))

        self.add(Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu"))
        self.add(Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu"))
        self.add(MaxPooling2D((2, 2)))

        self.add(Conv2D(256, kernel_size=(3, 3), padding="same", activation="relu"))
        self.add(Conv2D(256, kernel_size=(3, 3), padding="same", activation="relu"))
        self.add(Conv2D(256, kernel_size=(3, 3), padding="same", activation="relu"))
        self.add(MaxPooling2D((2, 2)))

        self.add(Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu"))
        self.add(Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu"))
        self.add(Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu"))
        self.add(MaxPooling2D((2, 2)))

        self.add(Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu"))
        self.add(Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu"))
        self.add(Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu"))
        self.add(MaxPooling2D((2, 2)))

        self.add(Flatten())
        self.add(Dense(4096, activation="relu"))
        self.add(Dense(4096, activation="relu"))
        self.add(Dense(classes, activation="softmax"))

class VGG_19(Sequential):
    def __init__(self, width, height, depth, classes=1000):
        super(VGG_19, self).__init__()

        input_shape = (height, width, depth)

        if tf.keras.backend.image_data_format() == "channels_first":
            input_shape = (depth, height, width)

        self.add(Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu", input_shape=input_shape))
        self.add(Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"))
        self.add(MaxPooling2D((2, 2)))

        self.add(Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu"))
        self.add(Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu"))
        self.add(MaxPooling2D((2, 2)))

        self.add(Conv2D(256, kernel_size=(3, 3), padding="same", activation="relu"))
        self.add(Conv2D(256, kernel_size=(3, 3), padding="same", activation="relu"))
        self.add(Conv2D(256, kernel_size=(3, 3), padding="same", activation="relu"))
        self.add(Conv2D(256, kernel_size=(3, 3), padding="same", activation="relu"))
        self.add(MaxPooling2D((2, 2)))

        self.add(Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu"))
        self.add(Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu"))
        self.add(Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu"))
        self.add(Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu"))
        self.add(MaxPooling2D((2, 2)))

        self.add(Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu"))
        self.add(Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu"))
        self.add(Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu"))
        self.add(Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu"))
        self.add(MaxPooling2D((2, 2)))

        self.add(Flatten())
        self.add(Dense(4096, activation="relu"))
        self.add(Dense(4096, activation="relu"))
        self.add(Dense(classes, activation="softmax"))