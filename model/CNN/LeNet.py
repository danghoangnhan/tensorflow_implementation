
"""
Paper:  Y. Lecun, L. Bottou, Y. Bengio and P. Haffner, "Gradient-based learning applied to document recognition,"
        in Proceedings of the IEEE, vol. 86, no. 11, pp. 2278-2324, Nov. 1998, 
doi: 10.1109/5.726791.
Maintainer: DanielDu
This code is a part of the implementation of the above paper.
Please refer to the link for more details on the methodology and findings.
"""
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