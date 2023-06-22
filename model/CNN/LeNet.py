from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,AvgPool2D
from tensorflow.keras.models import Model

# LeNet-1,LeNet-4,LeNet-5

class LeNet_1(Model):
    def __init__(self):
        super(LeNet_1, self).__init__()
        
        self.conv1 = Conv2D(4, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1))
        self.avgpool1 = AvgPool2D(pool_size=(2, 2))
        self.conv2 = Conv2D(8, kernel_size=(5, 5), activation='relu')
        self.avgpool2 = AvgPool2D(pool_size=(2, 2))
        self.flatten = Flatten()
        self.dense = Dense(10, activation='softmax')
        
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.avgpool1(x)
        x = self.conv2(x)
        x = self.avgpool2(x)
        x = self.flatten(x)
        output = self.dense(x)
        
        return output
    
class LeNet_4(Model):
    def __init__(self):
        super(LeNet_4, self).__init__()
        
        self.conv1 = Conv2D(4, kernel_size=(5, 5), activation='relu', input_shape=(32, 32, 1))
        self.avgpool1 = AvgPool2D(pool_size=(2, 2))
        self.conv2 = Conv2D(16, kernel_size=(5, 5), activation='relu')
        self.avgpool2 = AvgPool2D(pool_size=(2, 2))
        self.flatten = Flatten()
        self.dense1 = Dense(120, activation='relu')
        self.dense2 = Dense(10, activation='softmax')
        
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.avgpool1(x)
        x = self.conv2(x)
        x = self.avgpool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        output = self.dense2(x)
        
        return output
    
class LeNet5(Model):
    def __init__(self):
        super(LeNet5, self).__init__()
        #   Select 6 feature convolution kernels with a size of 5 * 5 (without offset), and get 66 feature maps. The size of each feature map is 32−5 + 1 = 2832−5 + 1 = 28.
        #   That is, the number of neurons has been reduced from 10241024 to 28 ∗ 28 = 784 28 ∗ 28 = 784.
        #   Parameters between input layer and C1 layer: 6 ∗ (5 ∗ 5 + 1)
        self.conv1 = Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1))
        # The input of this layer is the output of the first layer, which is a 28 * 28 * 6 node matrix.
        # The size of the filter used in this layer is 2 * 2, and the step length and width are both 2, 
        # so the output matrix size of this layer is 14 * 14 * 6.
        self.maxpool1 = MaxPooling2D(pool_size=(2, 2))
        #   The input matrix size of this layer is 14 * 14 * 6, the filter size used is 5 * 5, and the depth is 16.
        #   This layer does not use all 0 padding, and the step size is 1.
        #   The output matrix size of this layer is 10 * 10 * 16. This layer has 5 * 5 * 6 * 16 + 16 = 2416 parameters
        self.conv2 = Conv2D(16, kernel_size=(5, 5), activation='relu')
        #   The input matrix size of this layer is 10 * 10 * 16. The size of the filter used in this layer is 2 * 2, and the length and width steps are both 2,
        #   so the output matrix size of this layer is 5 * 5 * 16.
        self.maxpool2 = MaxPooling2D(pool_size=(2, 2))
        #   The input matrix size of this layer is 5 * 5 * 16.
        #   This layer is called a convolution layer in the LeNet-5 paper, but because the size of the filter is 5 * 5, #
        #   So it is not different from the fully connected layer. 
        #   If the nodes in the 5 * 5 * 16 matrix are pulled into a vector, then this layer is the same as the fully connected layer.
        #   The number of output nodes in this layer is 120, with a total of 5 * 5 * 16 * 120 + 120 = 48120 parameters.
        self.flatten = Flatten()
        self.dense1 = Dense(120, activation='relu')
        #   The number of input nodes in this layer is 120 and the number of output nodes is 84.
        #   The total parameter is 120 * 84 + 84 = 10164 (w + b)
        self.dense2 = Dense(84, activation='relu')
        #   The number of input nodes in this layer is 84 and the number of output nodes is 10.
        #   The total parameter is 84 * 10 + 10 = 850
        self.dense3 = Dense(10, activation='softmax')
    
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        output = self.dense3(x)
        return output