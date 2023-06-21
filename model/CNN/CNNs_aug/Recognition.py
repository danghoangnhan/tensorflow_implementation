# -*- coding: utf-8 -*-
"""
Created on Tue May 19 03:19:05 2020

@author: tuyen
"""

import cv2
# from lee.preprocessing import ImageToArrayPreprocessor
# from lee.preprocessing import SimplePreprocessor
# from lee.datasets import SimpleDatasetLoader
# from lee.nn.conv.shallownet import ShallowNet
from imutils import paths
from tensorflow.keras.models import load_model

# classLabels = ["cat", "dog", "panda"]

classLabels = ["bluebell", "buttercup", "colts'foot",
               "cowslip", "crocus", "daffodil", "daisy",
               "dandelion", "fritillary", "iris",
               "lilyvalley", "pansy", "snowdrop",
               "sunflower", "tigerlily", "tulip",
               "windflower"]

# grab the list of images in the dataset then randomly sample
# indexes into the image paths list
print("[INFO] sampling images...")
# magePaths = np.array(list(paths.list_images(args["dataset"])))
imagePaths = list(paths.list_images("D:/Hop tac voi Indonesia 2022/UNDIKSHA/Programs/Flowers/test"))  #
# idxs = np.random.randint(0, len(imagePaths), size=(15,))
# imagePaths = imagePaths[idxs]
# initialize the image preprocessors
# sp = SimplePreprocessor(299,299)
# For KdadexNET
# sp = SimplePreprocessor(32,32)
# For LeNet
sp = SimplePreprocessor(28, 28)
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixel intensities
# to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths)
data = data.astype("float") / 255.0
print(data.shape)
# load the pre-trained network
print("[INFO] loading pre-trained network...")

model = load_model("D:/Hop tac voi Indonesia 2022/UNDIKSHA/weights/myNET_weights.hdf5")
# make predictions on the images
print("[INFO] recognition...")
preds = model.predict(data, batch_size=32).argmax(axis=1)
# loop over the sample images
for (i, imagePath) in enumerate(imagePaths):
    # load the example image, draw the prediction, and display it
    # to our screen
    image = cv2.imread(imagePath)
    cv2.putText(image, "Label: {}".format(classLabels[preds[i]]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                2)
    cv2.imshow("Image", image)

    save_path = "D:/Hop tac voi Indonesia 2022/UNDIKSHA/results/" + str(i) + ".bmp"
    cv2.imwrite(save_path, image)
    # print("E:/MCUT/Neural Networks/Day 4/1.ShallowNet/recognition_results/"+str(i)+".bmp")
    cv2.waitKey(0)
