from tensorflow.keras.models import Model
from tensorflow.keras.layers import  MaxPooling2D,Conv2D,UpSampling2D,Layer,Input

import tensorflow as tf

# 編碼器(Encoder)
class Encoder(Layer):
    def __init__(self, filters):
        super(Encoder, self).__init__()
        self.conv1 = Conv2D(filters=filters[0], kernel_size=3, strides=1, activation='relu', padding='same')
        self.conv2 = Conv2D(filters=filters[1], kernel_size=3, strides=1, activation='relu', padding='same')
        self.conv3 = Conv2D(filters=filters[2], kernel_size=3, strides=1, activation='relu', padding='same')
        self.pool = MaxPooling2D((2, 2), padding='same')
               
    
    def call(self, input_features):
        x = self.conv1(input_features)
        #print("Ex1", x.shape)
        x = self.pool(x)
        #print("Ex2", x.shape)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        return x

# 解碼器(Decoder)
class Decoder(Layer):
    def __init__(self, filters):
        super(Decoder, self).__init__()
        self.conv1 = Conv2D(filters=filters[2], kernel_size=3, strides=1, activation='relu', padding='same')
        self.conv2 = Conv2D(filters=filters[1], kernel_size=3, strides=1, activation='relu', padding='same')
        self.conv3 = Conv2D(filters=filters[0], kernel_size=3, strides=1, activation='relu', padding='valid')
        self.conv4 = Conv2D(1, 3, 1, activation='sigmoid', padding='same')
        self.upsample = UpSampling2D((2, 2))
  
    def call(self, encoded):
        x = self.conv1(encoded)
        # 上採樣
        x = self.upsample(x)

        x = self.conv2(x)
        x = self.upsample(x)
        
        x = self.conv3(x)
        x = self.upsample(x)
        
        return self.conv4(x)
    
class Autoencoder(Model):
    def __init__(self, filters):
        super(Autoencoder, self).__init__()
        self.loss = []
        self.encoder = Encoder(filters)
        self.decoder = Decoder(filters)

    def call(self, input_features):
        #print(input_features.shape)
        encoded = self.encoder(input_features)
        #print(encoded.shape)
        reconstructed = self.decoder(encoded)
        #print(reconstructed.shape)
        return reconstructed
    

class Encoder(Layer):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')
        self.pool1 = MaxPooling2D((2, 2), padding='same')
        self.conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')
        self.pool2 = MaxPooling2D((2, 2), padding='same')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        encoded = self.pool2(x)
        return encoded
    
class BottleNeck(Layer):
    def __init__(self):
        super(BottleNeck, self).__init__()
        self.bottle_neck = Conv2D(256, (3, 3), activation='relu', padding='same')
        self.encoder_visualization = Conv2D(256, (3, 3), activation='relu', padding='same')
        
    def call(self, inputs):
        x = self.bottle_neck(inputs)
        x = self.encoder_visualization(x)
        return x
    
class Decoder(Layer):
    def __init__(self, input_shape, latent_dim):
        super(Decoder, self).__init__()
        self.conv1 = Conv2D(128, (3, 3), activation='relu', padding='same')
        self.up1 = UpSampling2D((2, 2))
        self.conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')
        self.up2 = UpSampling2D((2, 2))
        self.conv3 = Conv2D(1, (3, 3), activation='relu')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.up1(x)
        x = self.conv2(x)
        x = self.up2(x)
        decoded = self.conv3(x)
        return decoded

class ConvAutoencoder(tf.keras.Model):
    def __init__(self, input_shape, latent_dim):
        super(ConvAutoencoder, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(input_shape, latent_dim)
        self.bottle_neck = BottleNeck()
        
    def call(self, inputs):
        encoded = self.encoder(inputs)
        bottle_encoded = self.bottle_neck(encoded)
        decoded = self.decoder(bottle_encoded)
        return decoded

# Example usage
input_shape = (64, 64, 3)  # Example input shape for RGB images
latent_dim = 64  # Size of the latent representation
autoencoder_model = ConvAutoencoder(input_shape, latent_dim)
