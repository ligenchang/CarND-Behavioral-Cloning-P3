# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/center.jpg "Center"
[image3]: ./examples/turn.jpg "Turn"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with three 5x5 filter sizes, two 3x3 filter sizes and depths between 24 and 64 (model.py lines 94-108) 

The model includes RELU layers to introduce nonlinearity (code line 94 -113), and the data is normalized in the model using a Keras lambda layer with model.add(Lambda(lambda x: x /255 - 0.5, input_shape=(160,320,3)))(code line 87). 

#### 2. Attempts to reduce overfitting in the model

The model contains 5 dropout layers with dropout rate 0.25 in convolutional layers in order to reduce overfitting (model.py lines 94 -103). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 27 and 129). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 116).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. Also, initially the driving simulator not performed well on turning so that I specially captured more data for all turning corner to make it turn smoothly. Beside this , I also drive one round with clockwise to get more training data.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the Nvidia end-to-end self driving car CNN network architecture. I thought this model might be appropriate because it's specially designed for self drving car and can be used to real-time self driving car. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model by adding dropout layer in convolutional layer and full connected layer so that it will reduce the redundant data and beat the over-fit issue.

Then I realized that add dropout layer in full connected layer doesn't help on the performance in the simulator. According to the test result, I finally added 5 drop out layers to the convolutional layers only.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track on the turning corner. To improve the driving behavior in these cases, I manually record and add more training data for the turning corner case.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 94-113) consisted of a convolution neural network with the following layers and layer sizes:

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_11 (Lambda)           (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_6 (Cropping2D)    (None, 65, 320, 3)        0         
_________________________________________________________________
lambda_12 (Lambda) (Resize)          (None, 66, 200, 3)        0         
_________________________________________________________________
conv2d_26 (Conv2D)           (None, 31, 98, 24)        1824      
_________________________________________________________________
dropout_26 (Dropout)         (None, 31, 98, 24)        0         
_________________________________________________________________
conv2d_27 (Conv2D)           (None, 14, 47, 36)        21636     
_________________________________________________________________
dropout_27 (Dropout)         (None, 14, 47, 36)        0         
_________________________________________________________________
conv2d_28 (Conv2D)           (None, 5, 22, 48)         43248     
_________________________________________________________________
dropout_28 (Dropout)         (None, 5, 22, 48)         0         
_________________________________________________________________
conv2d_29 (Conv2D)           (None, 3, 20, 64)         27712     
_________________________________________________________________
dropout_29 (Dropout)         (None, 3, 20, 64)         0         
_________________________________________________________________
conv2d_30 (Conv2D)           (None, 1, 18, 64)         36928     
_________________________________________________________________
dropout_30 (Dropout)         (None, 1, 18, 64)         0         
_________________________________________________________________
flatten_6 (Flatten)          (None, 1152)              0         
_________________________________________________________________
dense_21 (Dense)             (None, 100)               115300    
_________________________________________________________________
dense_22 (Dense)             (None, 50)                5050      
_________________________________________________________________
dense_23 (Dense)             (None, 10)                510       
_________________________________________________________________
dense_24 (Dense)             (None, 1)                 11        
=================================================================
Total params: 252,219
Trainable params: 252,219
Non-trainable params: 0


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps (Clockwise and Anti-Clockwise) on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle in the turning corner so than the car will learn the correct behaviour when turning. This is the image captured for turning.

![alt text][image3]


Then I repeated this process on track two in order to get more data points.


For the flipping image technical, it doesn't work for me based on the test result as it added lots of noises to make the car go out of the track. I didn't dig it out too much why it can't work.

After the collection process, I had 8428 training data and 2107 validation data. I then preprocessed this data by utilizing left camera and right camera image as well and adjusted the angle with 0.23 so that the car know how go to center when it leans on the left or right.



I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used epochs as 100 and implemented early stopping with patience with 5. I set the samples_per_epoch= len(train_samples)/batch_size+1, which is different with the example given in the lesson. I used an adam optimizer so that manually training the learning rate wasn't necessary.
