# import the necessary packages
from keras import Input
from keras.layers import BatchNormalization, Flatten, AveragePooling2D
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.models import Model
from keras.optimizers import SGD
from keras.regularizers import l2
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# for ImageDataGenerator
from tensorflow import add
from tensorflow.python.keras.callbacks import LearningRateScheduler


class ResNet:
    @staticmethod
    def residual_module(data, K, stride, chanDim, red=False, reg=0.0001, bnEps=2e-5, bnMom=0.9):
        # the shortcut branch of the ResNet module should be initialize as the input (identity) data
        shortcut = data
        # the first block of the ResNet module are the 1x1 CONVs
        bn1 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(data)
        act1 = Activation("relu")(bn1)
        conv1 = Conv2D(int(K * 0.25), (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act1)
        # the second block of the ResNet module are the 3x3 CONVs
        bn2 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv1)
        act2 = Activation("relu")(bn2)
        conv2 = Conv2D(int(K * 0.25), (3, 3), strides=stride, padding="same", use_bias=False,
                       kernel_regularizer=l2(reg))(act2)
        # the third block of the ResNet module is another set of 1x1 CONVs
        bn3 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv2)
        act3 = Activation("relu")(bn3)
        conv3 = Conv2D(K, (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act3)
        # if we are to reduce the spatial size, apply a CONV layer to # the shortcut
        if red:
            shortcut = Conv2D(K, (1, 1), strides=stride, use_bias=False, kernel_regularizer=l2(reg))(act1)
        # add together the shortcut and the final CONV
        x = add([conv3, shortcut])
        # return the addition as the output of the ResNet module
        return x

    @staticmethod
    def build(width, height, depth, classes, stages, filters, reg=0.0001, bnEps=2e-5, bnMom=0.9):
        # initialize the input shape to be "channels last" and the channels dimension itself
        inputShape = (height, width, depth)
        chanDim = -1
        # if we are using "channels first", update the input shape and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
        # set the input and apply BN
        inputs = Input(shape=inputShape)
        x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(inputs)
        # loop over the number of stages
        for i in range(0, len(stages)):
            # initialize the stride, then apply a residual module used to reduce the spatial size of the input volume
            stride = (1, 1) if i == 0 else (2, 2)
            x = ResNet.residual_module(x, filters[i + 1], stride, chanDim, red=True, bnEps=bnEps, bnMom=bnMom)
            # loop over the number of layers in the stage
            for j in range(0, stages[i] - 1):
                # apply a ResNet module
                x = ResNet.residual_module(x, filters[i + 1], (1, 1), chanDim, bnEps=bnEps, bnMom=bnMom)
        # apply BN => ACT => POOL
        x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(x)
        x = Activation("relu")(x)
        x = AveragePooling2D((8, 8))(x)
        # softmax classifier
        x = Flatten()(x)
        x = Dense(classes, kernel_regularizer=l2(reg))(x)
        x = Activation("softmax")(x)
        # create the model
        model = Model(inputs, x, name="resnet")
        # return the constructed network architecture
        return model


######################## Adaptive learning rates #############################
# from keras.callbacks import LearningRateScheduler


def step_decay(epoch):
    # initialize the base initial learning rate, drop factor, and epochs to drop every
    initAlpha = 0.01
    factor = 0.25
    dropEvery = 5
    # compute learning rate for the current epoch
    alpha = initAlpha * (factor ** np.floor((1 + epoch) / dropEvery))
    # return the learning rate
    return float(alpha)


######################## Main Program ########################################
print("[INFO] loading images...")
path_to_dataset = "D:/MCUT/Neural Network/datasets/animals"
imagePaths = list(paths.list_images(path_to_dataset))
sp = SimplePreprocessor(32, 32, 3)
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
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")
# initialize the optimizer and model
print("[INFO] compiling model...")

no_epochs = 5
no_verbose = 100
no_batch_size = 32  # 32 images will be presented to the network at a time,
# and a full forward and backward pass will be
# done to update the parameters of the network
# initialize stochastic gradient descent with learning rate of 0.005
# how to tune learning rates ?????
#  without decay parameter
# opt = SGD(lr=0.005)
# opt = SGD(lr=0.005, momentum=0.9, nesterov=True)
#  with decay parameter
opt = SGD(lr=0.005, decay=0.01 / no_epochs, momentum=0.9, nesterov=True)
# Adaptive Learning Rates
callbacks = [LearningRateScheduler(step_decay)]
opt = SGD(lr=0.005, momentum=0.9, nesterov=True)
# opt = Adam(lr=0.001)
# Instantiate AlexNet architecture
# input image size 32x32
# output class is 3

model = ResNet.build(width=32, height=32, depth=3, classes=3, stages=(9, 9, 9), filters=(64, 64, 128, 256), reg=0.0005,
                     bnEps=2e-5, bnMom=0.9)
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
    callbacks=callbacks,  # Adaptive Learning Rates
    verbose=1)

# save the network to disk
print("[INFO] serializing network...")
model.save("D:/Hop tac voi Indonesia 2022/UNDIKSHA/weights/ResNet_weights.hdf5")
# predict the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=no_batch_size)

print(classification_report(
    testY.argmax(axis=1),
    predictions.argmax(axis=1),
    target_names=["cat", "dog", "panda"]
))

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
