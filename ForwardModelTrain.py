#!/usr/bin/env python

# Training the forward kinematic model of the Sawyer Robot

# import libraries
import os
import numpy as np
import math
#import rospy
#import roslib
import tensorflow
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dense
from keras import optimizers

# Define the path of the data
training_file='UniformedForwardModelData3.txt'

# Define where you want the model to be stored
model_file='UniformedForwardModel3.h5'

# Create the training dataset
training_data=np.loadtxt(training_file,delimiter=',')

# Dataset to learn the forward kinematics of the Sawyer Robot. Position Prediction only
input_training_data=np.delete(training_data,[2,4,6,7,8,9],axis=1)
output_training_data=np.delete(training_data,[0,1,2,3,4,5,6],axis=1)

# Define the Model
forwardModel=Sequential()

# Add the input Layer and the 1rst Hidden Layer
forwardModel.add(Dense(400,input_shape=(4,),init="uniform",activation="sigmoid"))

# Add the Hidden Layer
forwardModel.add(Dense(400,init="uniform",activation="sigmoid"))
forwardModel.add(Dense(300,init="uniform",activation="sigmoid"))
forwardModel.add(Dense(300,init="uniform",activation="sigmoid"))
forwardModel.add(Dense(200,init="uniform",activation="sigmoid"))

# Add the Output Layer
forwardModel.add(Dense(3))

# Define the Training Process
forwardModel.compile(optimizer='adam',loss='mse',metrics=['accuracy'])

# Train the Model
forwardModel.fit(input_training_data,output_training_data,validation_split=0.2,batch_size=1)

# Save the Model
forwardModel.save(model_file)






