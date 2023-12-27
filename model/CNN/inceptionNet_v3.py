import tensorflow as tf
from tensorflow.keras.layers.merge import concatenate
from tensorflow.keras.layers import Input, MaxPooling2D,  Dropout, GlobalAveragePooling2D, Dense,AveragePooling2D
from InceptionNet import ConvModule

class StemBlock(tf.keras.Sequential):
    def __init__(self):
        super(StemBlock,self).__init__()
        self.add(ConvModule(32, (3, 3),strides=2),padding='same')
        self.add(ConvModule(32, (3, 3),strides=1),padding='same')
        self.add(ConvModule(64, (3, 3),strides=1,padding='same'))
        self.add(MaxPooling2D((3,3),strides=2))
        self.add(ConvModule(80, (1, 1),strides=1),padding='same')
        self.add(ConvModule(192, (3, 3),strides=1),padding='same')
        self.add(MaxPooling2D((3,3),strides=2))

class InceptionBlock_A(tf.keras.layers.Layer):
    def __init__(self):
        super(InceptionBlock_A,self).__init__()
        self.branch_1 = tf.keras.Sequential([
            ConvModule(filters = 64 , kernel_size=(1, 1), padding='same'),
            ConvModule(filters = 96 , kernel_size=(3, 3), padding='same'),
            ConvModule(filters = 96 , kernel_size=(3, 3), padding='same'),
        ])
        self.branch_2 = tf.keras.Sequential([
            ConvModule(filters = 48 , kernel_size=(1, 1), padding='same'),
            ConvModule(filters = 64 , kernel_size=(3, 3), padding='same'),
        ])
        self.branch_3 = tf.keras.Sequential([
            AveragePooling2D(pool_size=(3,3) , strides=1 , padding='same'),        
            ConvModule(filters = 64 , kernel_size=(3, 3), padding='same'),
        ])
        self.branch_4 = ConvModule(filters = 64 , kernel_size=(1, 1))

    def call(self,inputs):
        branch1 = self.branch_1(inputs)
        branch2 = self.branch_2(inputs)
        branch3 = self.branch_3(inputs)
        branch4 = self.branch_4(inputs)
        out = concatenate([branch1 , branch2 , branch3,branch4] , axis=3)
        return out
    
class InceptionBlock_B(tf.keras.layers.Layer):
    def __init__(self,filters):
        super(InceptionBlock_B,self).__init__()
        self.branch_1 = tf.keras.Sequential([
            ConvModule(filters = filters , kernel_size=(1, 1), padding='same'),
            ConvModule(filters = filters , kernel_size=(7, 1), padding='same'),
            ConvModule(filters = filters , kernel_size=(1, 7), padding='same'),
            ConvModule(filters = filters , kernel_size=(7, 1), padding='same'),
            ConvModule(filters = 192     , kernel_size=(1, 7), padding='same'),
        ])
        self.branch_2 = tf.keras.Sequential([
            ConvModule(filters = filters , kernel_size=(1, 1), padding='same'),
            ConvModule(filters = filters , kernel_size=(1, 7), padding='same'),
            ConvModule(filters = 192     , kernel_size=(7, 1), padding='same'),
        ])
        self.branch_3 = tf.keras.Sequential([
            AveragePooling2D(pool_size=(3,3) , strides=1 , padding='same'),        
            ConvModule(filters = 64 , kernel_size=(3, 3), padding='same'),

        ])
        self.branch_4 = ConvModule(filters = 192 , kernel_size=(1, 1))

    def call(self,inputs):
        branch1 = self.branch_1(inputs)
        branch2 = self.branch_2(inputs)
        branch3 = self.branch_3(inputs)
        branch4 = self.branch_4(inputs)
        out = concatenate([branch1 , branch2 , branch3,branch4] , axis=3)
        return out  
    
class InceptionBlock_C(tf.keras.layers.Layer):
    def __init__(self):
        super(InceptionBlock_C,self).__init__()
        self.branch_1 = tf.keras.Sequential([
            ConvModule(filters = 448 , kernel_size=(1, 1), padding='same'),
            ConvModule(filters = 384 , kernel_size=(3, 3), padding='same'),
        ])
        self.branch_1_1 = ConvModule(filters = 384 , kernel_size=(1, 3), padding='same')
        self.branch_1_2 = ConvModule(filters = 384 , kernel_size=(3, 1), padding='same')

        self.branch_2 = ConvModule(filters = 384 , kernel_size=(1, 1), padding='same')
        self.branch_2_1 = ConvModule(filters = 384 , kernel_size=(1, 3), padding='same')
        self.branch_2_2 = ConvModule(filters = 384 , kernel_size=(3, 1), padding='same')


        self.branch_3 = tf.keras.Sequential([
            AveragePooling2D(pool_size=(3,3) , strides=1 , padding='same'),        
            ConvModule(filters = 192 , kernel_size=(3, 3), padding='same'),

        ])
        self.branch_4 = ConvModule(filters = 320 , kernel_size=(1, 1))

    def call(self,inputs):
        branch1 = self.branch_1(inputs)
        branch1 = concatenate([self.branch_1_1(branch1) , self.branch_1_2(branch1)], axis = 3)
        branch2 = self.branch_2(inputs)
        branch2 = concatenate([self.branch_2_1(branch2) , self.branch_2_2(branch2)], axis = 3)
        
        branch3 = self.branch_3(inputs)
        branch4 = self.branch_4(inputs)
        out = concatenate([branch1 , branch2 , branch3,branch4] , axis=3)
        return out
    
class ReductionBlock_A(tf.keras.layers.Layer):
    def __init__(self):
        super(ReductionBlock_A,self).__init__()
        self.branch_1 = tf.keras.Sequential([
                ConvModule(filters = 64 , kernel_size=(1, 1),strides=1, padding='same'),
                ConvModule(filters = 96 , kernel_size=(3, 3),strides=1, padding='same'),
                ConvModule(filters = 96 , kernel_size=(3, 3),strides=2, padding='same'),
        ])
        self.branch_2 = ConvModule(filters = 384, filter_Size=(3,3) , strides=2)
        
        self.branch_3 = MaxPooling2D(pool_size=(3,3) , strides=(2,2) , padding='same')
        
    def call(self,inputs):
        branch1 = self.branch_1(inputs)
        branch2 = self.branch_2(inputs)
        branch3 = self.branch_3(inputs)
        output = concatenate([branch1 , branch2 , branch3], axis = 3)
        return output
    
class ReductionBlock_B(tf.keras.layers.Layer):
    def __init__(self):
        super(ReductionBlock_B,self).__init__()
        self.branch_1 = tf.keras.Sequential([
                ConvModule(filters = 192 , kernel_size=(1, 1),strides=1, padding='same'),
                ConvModule(filters = 192 , kernel_size=(1, 7),strides=1, padding='same'),
                ConvModule(filters = 192 , kernel_size=(7, 1),strides=1, padding='same'),
                ConvModule(filters = 192 , kernel_size=(3, 3),strides=2, padding='valid'),
        ])
        self.branch_2 = tf.keras.Sequential([
                ConvModule(filters = 192 , kernel_size=(1, 1),strides=1, padding='same'),
                ConvModule(filters = 320 , kernel_size=(3, 3),strides=2, padding='valid'),
        ])        
        self.branch_3 = MaxPooling2D(pool_size=(3,3) , strides=(2,2) , padding='same')
        
    def call(self,inputs):
        branch1 = self.branch_1(inputs)
        branch2 = self.branch_2(inputs)
        branch3 = self.branch_3(inputs)
        output = concatenate([branch1 , branch2 , branch3], axis = 3)
        return output

class InceptionV3(tf.keras.Model):
    def __init__(self,input_shape,num_class):
        super(InceptionV3,self).__init__()
        self.input_layer = Input(shape=input_shape)    
        self.stemBlock = StemBlock()
        self.inception_A_Blocks = [InceptionBlock_A(filters=element) for element in [32, 64, 64]]
        self.inception_B_Blocks = [InceptionBlock_B(filters=element) for element in [128, 160, 160, 192]]
        self.inception_C_Blocks = [InceptionBlock_C() for _ in range(2)]
        
        self.reductionBlock_A = ReductionBlock_A()
        self.reductionBlock_B = ReductionBlock_B()


        self.fc = tf.keras.Sequential([
            InceptionBlock_C(),
            InceptionBlock_C(),
            GlobalAveragePooling2D(),
            Dense(units = 2048 , activation='relu'),
            Dropout(0,2),
            Dense(units = num_class ,activation='softmax')
        ])
        
    def call(self,inputs):
        out = self.input_layer(inputs)
        out = self.stemBlock(out)
        for block in  self.inception_A_Blocks:
            out = block(out)
        out = self.reductionBlock_A(out)

        for block in  self.inception_B_Blocks:
            out = block(out)
        for block in  self.inception_C_Blocks:
            out = block(out)
        out = self.reductionBlock_B(out)


        out = self.fc(out)
        return out