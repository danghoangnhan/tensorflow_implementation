import tensorflow as tf
from tensorflow.keras.layers.merge import concatenate
from tensorflow.keras.layers import Input, MaxPooling2D, Dense, Flatten
from InceptionNet import ConvModule

class StemBlock(tf.keras.Sequential):
    def __init__(self):
        super(StemBlock,self).__init__()
        self.add(ConvModule(32, (3, 3),strides=2))
        self.add(ConvModule(32, (3, 3),strides=1))
        self.add(ConvModule(32, (3, 3),strides=1,padding='same'))
        self.add(ConvModule(80, (1, 1),strides=1))
        self.add(ConvModule(192, (3, 3),strides=1))
        self.add(MaxPooling2D((3,3),strides=2))

class InceptionBlock(tf.keras.layers.Layer):
    def __init__(self):
        super(InceptionBlock,self).__init__()
        self.branch_1 = tf.keras.Sequential([
            ConvModule(filters = 64 , kernel_size=(1, 1)),
            ConvModule(filters = 64 , kernel_size=(3, 3), padding='same'),
            ConvModule(filters = 96 , kernel_size=(3, 3), padding='same'),
        ])
        self.branch_2 = tf.keras.Sequential([
            ConvModule(filters = 48 , kernel_size=(1, 1)),
            ConvModule(filters = 64 , kernel_size=(5, 5), padding='same'),
        ])
        self.branch_3 = ConvModule(filters = 96 , kernel_size=(1, 1))

        self.branch_4 = tf.keras.Sequential([
            MaxPooling2D((3,3) , strides=1 , padding='same'),
            ConvModule(filter= 64 , kernel_size=(1, 1)),
        ])
    def call(self,inputs):
        branch1 = self.branch_1(inputs)
        branch2 = self.branch_2(inputs)
        branch3 = self.branch_3(inputs)
        branch4 = self.branch_4(inputs)
        return concatenate([branch1 , branch2 , branch3 , branch4] , axis=3)


class InceptionRestNetBlock_A(tf.keras.layers.Layer):
    def __init__(self):
        super(InceptionRestNetBlock_A,self).__init__()
        self.branch_1 = tf.keras.Sequential([
            ConvModule(filters = 32 , kernel_size=(1, 1)),
            ConvModule(filters = 48 , kernel_size=(3, 3), padding='same'),
            ConvModule(filters = 96 , kernel_size=(3, 3), padding='same'),
        ])
        self.branch_2 = tf.keras.Sequential([
            ConvModule(filters = 32 , kernel_size=(1, 1)),
            ConvModule(filters = 32 , kernel_size=(5, 5), padding='same'),
        ])
        self.branch_3 = ConvModule(filters = 32 , kernel_size=(1, 1))

        self.bottelNeck = ConvModule(filters = 320 , kernel_size=(1, 1))

    def call(self,inputs):
        branch1 = self.branch_1(inputs)
        branch2 = self.branch_2(inputs)
        branch3 = self.branch_3(inputs)
        out = concatenate([branch1 , branch2 , branch3] , axis=3)
        out = self.bottelNeck(out)
        out += inputs
        return out


class InceptionRestNetBlock_B(tf.keras.layers.Layer):
    def __init__(self):
        super(InceptionRestNetBlock_B,self).__init__()
        self.branch_1 = tf.keras.Sequential([
            ConvModule(filters = 128 , kernel_size=(1, 1)),
            ConvModule(filters = 160 , kernel_size=(1, 7), padding='same'),
            ConvModule(filters = 192 , kernel_size=(7, 1), padding='same'),
        ])
        self.branch_2 = ConvModule(filters = 192 , kernel_size=(1, 1))

        self.bottelNeck = ConvModule(filters = 2080 , kernel_size=(1, 1))

    def call(self,inputs):
        branch1 = self.branch_1(inputs)
        branch2 = self.branch_2(inputs)
        out = concatenate([branch1 , branch2] , axis=3)
        out = self.bottelNeck(out)
        out += inputs
        return out

class InceptionRestNetBlock_C(tf.keras.layers.Layer):
    def __init__(self):
        super(InceptionRestNetBlock_B,self).__init__()
        self.branch_1 = tf.keras.Sequential([
            ConvModule(filters = 192 , kernel_size=(1, 1)),
            ConvModule(filters = 224 , kernel_size=(1, 3), padding='same'),
            ConvModule(filters = 256 , kernel_size=(3, 1), padding='same'),
        ])
        self.branch_2 = ConvModule(filters = 192 , kernel_size=(1, 1))

        self.bottelNeck = ConvModule(filters = 2080 , kernel_size=(1, 1))

    def call(self,inputs):
        branch1 = self.branch_1(inputs)
        branch2 = self.branch_2(inputs)
        out = concatenate([branch1 , branch2] , axis=3)
        out = self.bottelNeck(out)
        out += inputs
        return out


class Reduction_A(tf.keras.layers.Layer):
    def __init__(self):
        super(InceptionRestNetBlock_C,self).__init__()
        self.branch_1 = tf.keras.Sequential([
            ConvModule(filters=256,kernel_size=(1, 1),strides=1),
            ConvModule(filters=256,kernel_size=(3, 3),strides=1, padding='same'),
            ConvModule(filters=384,kernel_size=(3, 3),strides=2),
        ])
        self.branch_2 = ConvModule(filters = 384 , kernel_size=(3, 3),strides=2)
        self.branch_3 = MaxPooling2D((3, 3),strides=2)

    def call(self,inputs):
        branch1 = self.branch_1(inputs)
        branch2 = self.branch_2(inputs)
        branch3 = self.branch_3(inputs)

        out = concatenate([branch1 , branch2,branch3] , axis=3)
        return out
    
class Reduction_B(tf.keras.layers.Layer):
    def __init__(self):
        super(InceptionRestNetBlock_C,self).__init__()
        self.branch_1 = tf.keras.Sequential([
            ConvModule(filters=256,kernel_size=(1, 1),strides=1),
            ConvModule(filters=288,kernel_size=(3, 3),strides=1, padding='same'),
            ConvModule(filters=320,kernel_size=(3, 3),strides=2),
        ])
        self.branch_2 = tf.keras.Sequential([
            ConvModule(filters=256,kernel_size=(1, 1),strides=1),
            ConvModule(filters=384,kernel_size=(3, 3),strides=2),
        ])
        self.branch_3 = tf.keras.Sequential([
            ConvModule(filters=256,kernel_size=(1, 1),strides=1),
            ConvModule(filters=288,kernel_size=(3, 3),strides=2),
        ])
        self.branch_4 = MaxPooling2D((3, 3),strides=2)

    def call(self,inputs):
        branch1 = self.branch_1(inputs)
        branch2 = self.branch_2(inputs)
        branch3 = self.branch_3(inputs)
        branch4 = self.branch_4(inputs)

        out = concatenate([branch1 , branch2, branch3, branch4] , axis=3)
        return out

class InceptionV2(tf.keras.Model):
    def __init__(self,input_shape,num_class):
        super(InceptionV2,self).__init__()
        self.input_layer = Input(shape=input_shape)    
        self.stemBlock = StemBlock()
        self.inceptionBlock = InceptionBlock()
        self.inceptionResNet_A_Blocks = [InceptionRestNetBlock_A() for _ in range(10)]
        self.reduction_a = Reduction_A()
        self.inceptionResNet_B_Blocks = [InceptionRestNetBlock_B() for _ in range(10)]
        self.reduction_b = Reduction_B()
        self.inceptionResNet_C_Blocks = [InceptionRestNetBlock_C() for _ in range(10)]
        
        self.fc = tf.keras.Sequential([
            ConvModule(filters=1536,kernel_size=(1, 1),strides=1),
            ConvModule(filters=1536,kernel_size=(8, 8),strides=1),
            Flatten(),
            Dense(units = 1536 , activation='relu'),
            Dense(units = num_class ,activation='softmax')
        ])
        
    def call(self,inputs):
        out = self.input_layer(inputs)
        out = self.stemBlock(out)
        out = self.inceptionBlock(out)
        for block in  self.inceptionResNet_A_Blocks:
            out = block(out)
        out = self.reduction_a(out)
        for block in  self.inceptionResNet_B_Blocks:
            out = block(out)
        out = self.reduction_b(out)
        for block in  self.inceptionResNet_C_Blocks:
            out = block(out)
        out = self.fc(out)
        return out