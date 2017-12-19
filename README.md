# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/center.jpg "Center"
[image3]: ./examples/turn.jpg "Turn"

Overview
---
This repository contains all files submitted for the Behavioral Cloning Project.
* model.py (script used to create and train the model)
* drive.py (script to drive the car - feel free to modify this file)
* model.h5 (a trained Keras model)
* a report writeup file (either markdown or pdf)
* video.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)

This prject gives one simulator which has two tracks to record the images and steering angles. We could use deep neural network and convolutional network to predicate the steering angle based on the input image. It should be various convolutional layers and several full connected layers and finally the output should be one value (dense 1) to generate the predicated angle.

Apparently, this should be a non-linear regression issue. In the csv file, we have center image, left camera image and right camera image. In the drive.py, it will connect to the simulator and get the real-time driving center image and input and get one predicated steering angle and based on that the simulator will clone the behaviour human was driving.I use Keras for the model training. In the backend, it's using tensorflow.

It's suggested  to train the model in one GPU machine with Nvidia diplay driver and run the simator in local machine to test the result. Do rememer that you need to make sure the Keras, Tensorflow version the same in training server and local machine, otherwise it will have error.



Model Architecture Description
---
There are several architecture we can choose to traing this model. E.g LeNET, Nvidia self driving car CNN, VGG etc. As Nvidia CNN architecture is specially designed for self driving car and can be used for real time image input and predication in real world self driving, so it should be our first choice.

In principle, we can just copy exactly the Nvidia CNN network, which can be found here: https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/

We can also base on this architecture to do some custimization. For me I tried to redue the convolutional layers from 5 to 3, changed the filter size etc and find it doesn't help and it even made the result become worse. After several days' testing, I finally choosed the original Nvidia CNN architecture but add dropout with 0.25 for each convolutional layer. For the full connect layer, the first layer with dense 1164 doesn't help too much. After removing the first full connect layer with dense 1164, the driving bahavior will be even slightly smooth and the model.h5 will be reduced from 19mb to 3 mb.

Here is the model description I used for this project. So the full model including one normalization layer, one cropping layer, one resizing layer and the original Nvidia CNN layers. After cropping, our image size is 65*32*3, but the Nvidia CNN image size is 66*200*3 so that we need to resize the image.


_________________________________________________________________
Layer (type)                 Output Shape              Param #   
________________________________________________________________
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 65, 320, 3)        0         
_________________________________________________________________
lambda_2 (Lambda)            (None, 66, 200, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 31, 98, 24)        1824      
_________________________________________________________________
dropout_1 (Dropout)          (None, 31, 98, 24)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 14, 47, 36)        21636     
_________________________________________________________________
dropout_2 (Dropout)          (None, 14, 47, 36)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 5, 22, 48)         43248     
_________________________________________________________________
dropout_3 (Dropout)          (None, 5, 22, 48)         0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 20, 64)         27712     
_________________________________________________________________
dropout_4 (Dropout)          (None, 3, 20, 64)         0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 18, 64)         36928     
_________________________________________________________________
dropout_5 (Dropout)          (None, 1, 18, 64)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 1152)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 1164)              1342092   
_________________________________________________________________
dense_2 (Dense)              (None, 100)               116500    
_________________________________________________________________
dense_3 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_4 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_5 (Dense)              (None, 1)                 11        
________________________________________________________________
Total params: 1,595,511
Trainable params: 1,595,511
Non-trainable params: 0

Training data preparation
---
Training data preparation is not easy. However, it's critical for this project. Remember to find one good laptop or PC to run the simulator to make sure the dring is smoothing without stuck. Here is my training data acquiring strategy:
* I tried the example data provided by this project and it can't work so I abandoned it
* Fristly, I drive one lap with anti-clockwise at average speed 4-6 mph and tested it and realized the data is not enough. The car is easily went out of the track.
* Then I drive another lap with clockwise and merged the data. I find that the result improved a lot. It only went out of track sometimes in the turning corner. 
* Then I decided to add more traing data for the turning point only. After that, the car seems can run in the track, but sometimes it still went to the side and not in the middle.

Here is an example image of center lane driving:

![alt text][image2]

This is the image captured for turning.

![alt text][image3]


Data augmentation
---
* I utilized the left camera image and right camera image and adjust the angle with +- 0.23 to keep the car in the middle of the road
* I also tried to flip all training images and set the angle to negative, but this doesn't help and even seems added noises to the network. So I abandoned this data augment 
* I tried to rotated the center image with 5 degree and adjusted the angle accordingly for data augmentation. It helps in the straight road but it makes the driving becomes worse in the turning points. I abandoned this as well.

Keras Generator
---
If we pre-process all images directly, it will consume a lot memory and will be a limitation if we need tons of images ot train. The generator is here to help. However, to understan how Keras generator works is not easy as the generator is not that straightforward.

Basically the generator used the static variable internally and it will remember the last execution status and will continue from the status from last run. So if we pass the trainin samples via model.fit_generator, the generator will yield one array with length batch_size. Let's say we have 10 training examples and batch size is 2, then it will yield 5 times to go though all data samples. Based on this, we should set the samples_per_epoch= len(train_samples)/batch_size+1, which is different in the lesson's example and it will make training more faster and have more chances to get a better trained model and prevent the over-fitting.


Summary
---
I enjoyed a lot for training the network with Keras and it enhanced my understand of neural network. Based on the experience in track 1, I just recorded one time for track 2 and tested it and it magicly works, which is beyond my expectation. In traing the network, to prevent the overfitting is quite important, I find it will be useful to apply dropout in convolutional layer. Early stopping also quite helpful for helping to select the best performance model.

Appreciate sincerely to create such an interesting project for me to practice and it makes me addicated to the self driving.

