from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K


class LeNet:

    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
        # First hidden layer
        model.add(Conv2D(
            20,
            kernel_size=(5, 5),
            padding="same",
            input_shape=inputShape
        ))
        model.add(Activation("relu"))

        # Second hidden layer
        model.add(MaxPooling2D(strides=(2, 2)))

        # Third hidden layer
        model.add(Conv2D(
            50,
            kernel_size=(5, 5),
            padding="same"
        ))
        model.add(Activation("relu"))
        # Fourd hidden layer
        model.add(MaxPooling2D(strides=(2, 2)))

        # Outpt hidden layer
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model


######################## Adaptive learning rates #############################
# from keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import LearningRateScheduler


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
sp = SimplePreprocessor(28, 28, 3)
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
# opt = SGD(lr=0.005)
# opt = SGD(lr=0.005, momentum=0.9, nesterov=True)
#  with decay parameter
opt = SGD(lr=0.005, decay=0.01 / no_epochs, momentum=0.9, nesterov=True)
# Adaptive Learning Rates
callbacks = [LearningRateScheduler(step_decay)]
# opt = SGD(lr=0.005, momentum=0.9, nesterov=True)
# Instantiate ShallowNet architecture
# input image size 32x32
# output class is 3
model = LeNet.build(width=28, height=28, depth=3, classes=3)
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
#               callbacks=callbacks, # Adaptive Learning Rates
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
    callbacks=callbacks,
    verbose=1)

# save the network to disk
print("[INFO] serializing network...")
model.save("D:/Hop tac voi Indonesia 2022/UNDIKSHA/weights/LeNet_weights.hdf5")
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
