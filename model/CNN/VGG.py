from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
#source: https://www.kaggle.com/code/blurredmachine/vggnet-16-architecture-a-complete-guide?scriptVersionId=39674893
class VGG_16(Model):
    def __init__(self):
        super(VGG_16, self).__init__()
        self.conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu")
        self.conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu")
        self.pool1 = MaxPooling2D((2, 2))
        
        self.conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")
        self.conv4 = Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")
        self.pool2 = MaxPooling2D((2, 2))
        
        self.conv5 = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")
        self.conv6 = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")
        self.conv7 = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")
        self.pool3 = MaxPooling2D((2, 2))
        
        self.conv8 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")
        self.conv9 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")
        self.conv10 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")
        self.pool4 = MaxPooling2D((2, 2))
        
        self.conv11 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")
        self.conv12 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")
        self.conv13 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")
        self.pool5 = MaxPooling2D((2, 2))
        
        self.flatten = Flatten()
        self.dense1 = Dense(4096, activation="relu")
        self.dense2 = Dense(4096, activation="relu")
        self.output_layer = Dense(1000, activation="softmax")
    
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.pool1(x)
        
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.pool4(x)
        
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.pool5(x)
        
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        output = self.output_layer(x)
        
        return output

vgg16_model = VGG16Model()
