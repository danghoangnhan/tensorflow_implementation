# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 15:25:43 2021

"""
import numpy as np
from imutils import paths
# for ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator
#########################ImageToArrayPreprocessor##############################
# imports the img_to_array function from Keras
from keras_preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
# pip install imutils
from sklearn.preprocessing import LabelBinarizer


class ImageToArrayPreprocessor:
    def __init__(self, dataFormat=None):
        # store the image data format
        self.dataFormat = dataFormat

    def preprocess(self, image):
        # apply the the Keras utility function that correctly rearranges
        # the dimensions of the image
        return img_to_array(image, data_format=self.dataFormat)


########################SimplePreprocessor####################################
import cv2


class SimplePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        # store the target image width, height, and interpolation
        # method used when resizing
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        # resize the image to a fixed size, ignoring the aspect
        # ratio
        return cv2.resize(image, (self.width, self.height),
                          interpolation=self.inter)


########################SimpleDatasetLoader###################################
# import numpy as np
# import cv2
import os


class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        # store the image preprocessor
        self.preprocessors = preprocessors

        # if the preprocessors are None, initialize them as an
        # empty list
        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, imagePaths, verbose=-1):
        # initialize the list of features and labels
        data = []
        labels = []

        # loop over the input images
        for (i, imagePath) in enumerate(imagePaths):
            # load the image and extract the class label assuming
            # that our path has the following format:
            # /path/to/dataset/{class}/{image}.jpg
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]

            # check to see if our preprocessors are not None
            if self.preprocessors is not None:
                # loop over the preprocessors and apply each to
                # the image
                for p in self.preprocessors:
                    image = p.preprocess(image)

            # treat our processed image as a "feature vector"
            # by updating the data list followed by the labels
            data.append(image)
            labels.append(label)

            # show an update every `verbose` images
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1,
                                                      len(imagePaths)))

        # return a tuple of the data and labels
        return (np.array(data), np.array(labels))


#############################ShallowNet#######################################
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
from keras.optimizers import SGD


# Input => CONV => RELU => FC
class ShallowNet:

    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model along with the input shape to be "channels last"
        model = Sequential()
        inputShape = (height, width, depth)

        # if we are using "channels first", update the inputShape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        # define the first (and only) CONV => RELU layer
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))

        # softmax classifier
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        # return the constructed network architecture
        return model


######################## Adaptive learning rates #############################


def step_decay(epoch):
    # initialize the base initial learning rate, drop factor, and epochs to drop every
    initAlpha = 0.001
    factor = 0.75
    dropEvery = 10
    # compute learning rate for the current epoch
    alpha = initAlpha * (factor ** np.floor((1 + epoch) / dropEvery))
    # return the learning rate
    return float(alpha)


######################## Main Program ########################################

print("[INFO] loading images...")
path_to_dataset = "D:/MCUT/Neural Network/datasets/animals/train"
imagePaths = list(paths.list_images(path_to_dataset))
sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()

sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
# convert values to between 0-1
data = data.astype("float") / 255.0

# partition our data into training and test sets
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25,
                                                  random_state=42)

# convert the labels from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

# initialize the optimizer and model
print("[INFO] compiling model...")
no_epochs = 50
no_verbose = 1
no_batch_size = 32  # 32 images will be presented to the network at a time,
# and a full forward and backward pass will be
# done to update the parameters of the network
# initialize stochastic gradient descent with learning rate of 0.005
# how to tune learning rates ?????
# define the set of callbacks to be passed to the model during training

#  without decay parameter
opt = SGD(lr=0.005)
# opt = SGD(lr=0.005, momentum=0.9, nesterov=True)
#  with decay parameter
# opt = SGD(lr=0.001, decay=0.01 / no_epochs, momentum=0.9, nesterov=True)
# Adaptive Learning Rates
# callbacks = [LearningRateScheduler(step_decay)]
# opt = SGD(lr=0.001, momentum=0.9, nesterov=True) 
# Instantiate ShallowNet architecture
# input image size 32x32
# output class is 3
model = ShallowNet.build(width=32, height=32, depth=3, classes=3)
model.summary()
# compile the model
# loss function: cross-entropy and optimizer: SGD
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])
# train the network


print("[INFO] training network...")
# withouth data augmentation
# H = model.fit(trainX, trainY, validation_data=(testX, testY), 
#               batch_size=no_batch_size,
#               epochs=no_epochs, 
#               callbacks=callbacks,
#               verbose=no_verbose)

# data augmentation
# https://www.pyimagesearch.com/2018/12/24/how-to-use-keras-fit-and-fit_generator-a-hands-on-tutorial/
# for ImageDataGenerator
# from keras.preprocessing.image import ImageDataGenerator
H = model.fit_generator(
    aug.flow(trainX, trainY, batch_size=no_batch_size),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // 32,
    epochs=no_epochs,
    # callbacks=callbacks, # Adaptive Learning Rates
    verbose=1)

# save the network to disk
print("[INFO] serializing network...")
model.save("D:/Hop tac voi Indonesia 2022/UNDIKSHA/weights/shallownet_weights_new.hdf5")

# predict the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=no_batch_size)
# Show results
from sklearn.metrics import classification_report

print(classification_report(
    testY.argmax(axis=1),
    predictions.argmax(axis=1),
    target_names=["cat", "dog", "panda"]
))

# Draw results
import matplotlib.pyplot as plt

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, no_epochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, no_epochs), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, no_epochs), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, no_epochs), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()

##############################################################################
