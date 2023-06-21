# -*- coding: utf-8 -*-
"""
Created on Mon May 25 14:28:04 2020

@author: Lee
"""

# import the necessary packages
from keras.applications import VGG16
from keras.applications import ResNet152V2
from keras.applications import InceptionV3
from keras.applications import NASNetLarge

# load the VGG16 network
print("[INFO] loading network...")
# model = VGG16(weights="imagenet", include_top = False)
model = ResNet152V2(weights="imagenet", include_top=False)
# model = InceptionV3(weights="imagenet", include_top = False)

print("[INFO] showing layers...")

# loop over the layers in the network and display them to the
# console
for (i, layer) in enumerate(model.layers):
    print("[INFO] {}\t{}".format(i, layer.__class__.__name__))
