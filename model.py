import os
import csv
from sklearn.utils import shuffle
from scipy.ndimage import rotate
import cv2
import numpy as np
import sklearn
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Reshape
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras.backend import tf as ktf
import matplotlib.image as mpimg

#Read csv file to get all data
samples = []
with open('./img10/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

#split trainging samples and validation samples
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
train_samples = sklearn.utils.shuffle(train_samples)
validation_samples = sklearn.utils.shuffle(validation_samples)
iglobal=0 # use this for debugging to see how many time generator yields
def generator(samples, batch_size=32, validation_flag=True):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            global iglobal
            iglobal=iglobal+1
            #debug to see how many times yields
            #print ("yield", iglobal, "times")
            for batch_sample in batch_samples:
                for i in range(3):
                    name = './img10/IMG/'+batch_sample[i].split('/')[-1]
                    #image = cv2.imread(name)
                    image=mpimg.imread(name)
                    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    #image = exposure.equalize_hist(image)
                    angle = float(batch_sample[3])
                    if i==1: #left camera images 
                        angle=angle+0.23
                    if i==2: #right camera images
                        angle=angle-0.23
                        
                    
                    images.append(image)
                    angles.append(angle)
                    
                    if i==9: # this won't be executed as I find the flip image will add lots of noises
                        images.append(np.fliplr(image))
                        angles.append(angle*-1)
                    
                    if i==9: #this won't be exucuted das I can't find proper angle adjustment for rorate images
                        Rotate_image = rotate(image, 5, reshape=False) 
                        images.append(Rotate_image)
                        angles.append(angle-0.2)
                        Rotate_image = rotate(image, -5, reshape=False) 
                        images.append(Rotate_image)
                        angles.append(angle+0.2)
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function, select batch size with 16. If set it with 32, sometimes will encounter OOM exception
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)
print(len(train_samples))
print(len(validation_samples))
def resize_image(image):
    # In Nvidia CNN, the image input size is 66 x 200
    from keras.backend import tf as ktf   
    resized_image = ktf.image.resize_images(image, (66, 200))
    return resized_image

model=Sequential()
#nomalized images
model.add(Lambda(lambda x: x /255 - 0.5, input_shape=(160,320,3)))
#Cropping the image to only see the middle part and remove the noise
model.add(Cropping2D(cropping=((70,25), (0,0))))
#REsize the image to Nvidia input size
model.add(Lambda(resize_image, input_shape=(65,320,3), output_shape=(66, 200, 3)))

#Nvidia 5 lays of CNN with dropout 0.25 on each layer
model.add(Convolution2D(24,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
model.add(Dropout(0.25))
model.add(Convolution2D(36,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
model.add(Dropout(0.25))
model.add(Convolution2D(48,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
model.add(Dropout(0.25))
model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
model.add(Dropout(0.25))
model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(1164, activation='relu'))


model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

print(model.summary())

#use adam optimizer
model.compile(loss='mse', optimizer='adam')


#stop early to prevent overfitting, the tolerance is 2 epoches
filepath="model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# check 2 epochs patience
early_stop = EarlyStopping(monitor='val_loss',  patience=3, verbose=1, mode='min') 

callbacks_list = [checkpoint, early_stop]

#I set samples_per_epoch as len(train_samples)/batch_size +1 as I think for each epoch, looping through the whole training set is enough
model.fit_generator(train_generator, samples_per_epoch= len(train_samples)/32+1, validation_data=validation_generator, nb_val_samples=len(validation_samples)/32+1, nb_epoch=100,callbacks=callbacks_list, verbose=1)
#model.save('model.h5')
print ("model was saved successfully")