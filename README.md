# Tensorflow model Library

This is my personal library that implements  state-of-the-art deeplearning models. The models are implemented using TensorFlow and Keras.

## Requirements

- ![Conda](https://img.shields.io/badge/Conda-4.13.0-brightgreen)
- [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8.0-blue?logo=tensorflow)](https://tensorflow.org/)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/danghoangnhan/machineLearning
   cd machineLearning
   ```


## References


| Model                   | Task                    | Title                                                        | Authors                                              | Year | Link to Paper                                           |
|-------------------------|-------------------------|--------------------------------------------------------------|------------------------------------------------------|------|---------------------------------------------------------|
| LeNet (1, 4, 5)         | Image Classification    | Gradient-Based Learning Applied to Document Recognition     | Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner | 1998 | [Link](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) |
| AlexNet                 | Image Classification    | ImageNet Classification with Deep Convolutional Neural Networks | Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton    | 2012 | [Link](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) |
| VGG (16, 19)            | Image Classification    | Very Deep Convolutional Networks for Large-Scale Image Recognition | Karen Simonyan, Andrew Zisserman                      | 2014 | [Link](https://arxiv.org/abs/1409.1556)                  |
| ResNet (18, 34, 50, 101, 152) | Image Classification | Deep Residual Learning for Image Recognition                | Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun      | 2015 | [Link](https://arxiv.org/abs/1512.03385)                |
| InceptionV1 (GoogLeNet) | Image Classification    | Going Deeper with Convolutions                              | Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich | 2014 | [Link](https://arxiv.org/abs/1409.4842)                 |
| UNet                    | Semantic Image Segmentation | U-Net: Convolutional Networks for Biomedical Image Segmentation | Olaf Ronneberger, Philipp Fischer, Thomas Brox         | 2015 | [Link](https://arxiv.org/abs/1505.04597)                |