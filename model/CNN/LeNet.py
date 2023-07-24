from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,AvgPool2D
from tensorflow.keras.models import Sequential

class LeNet_1(Sequential):
    def __init__(self,classes=10):
        super(LeNet_1, self).__init__()
        self.add(Conv2D(4, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)))
        self.add(AvgPool2D(pool_size=(2, 2)))
        self.add(Conv2D(8, kernel_size=(5, 5), activation='relu'))
        self.add(AvgPool2D(pool_size=(2, 2)))
        self.add(Flatten())
        self.add(Dense(classes, activation='softmax'))

class LeNet_4(Sequential):
    def __init__(self,classes=10):
        super(LeNet_4, self).__init__()
        self.add(Conv2D(4, kernel_size=(5, 5), activation='relu', input_shape=(32, 32, 1)))
        self.add(AvgPool2D(pool_size=(2, 2)))
        self.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))
        self.add(AvgPool2D(pool_size=(2, 2)))
        self.add(Flatten())
        self.add(Dense(120, activation='relu'))
        self.add(Dense(classes, activation='softmax'))
    
class LeNet_5(Sequential):
    def __init__(self,classes=10):
        super(LeNet_5, self).__init__()
        
        self.add(Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)))
        self.add(MaxPooling2D(pool_size=(2, 2)))
        self.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))
        self.add(MaxPooling2D(pool_size=(2, 2)))
        self.add(Flatten())
        self.add(Dense(120, activation='relu'))
        self.add(Dense(84, activation='relu'))
        self.add(Dense(classes, activation='softmax'))