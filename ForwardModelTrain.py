#!/usr/bin/env python

# Training the forward kinematic model of the Sawyer Robot
import sys
import numpy as np
import tensorflow
from tensorflow.python.client import device_lib
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dense
from keras import optimizers
from keras.utils import multi_gpu_model

num_gpus = len(device_lib.list_local_devices()) - 1 
if len(sys.argv) > 2:
    num_gpus = min(int(sys.argv[1]), num_gpus)

# Define the path of the data
training_file='data/ufmdata_1.txt'

# Define where you want the model to be stored
model_file='models/forwardmodel.h5'

# Dataset to learn the forward kinematics of the Sawyer Robot. Position Prediction only
training_data=np.loadtxt(training_file,delimiter=',')
input_training_data=np.delete(training_data,[2,4,6,7,8,9],axis=1)
output_training_data=np.delete(training_data,[0,1,2,3,4,5,6],axis=1)

# Define the Model
forwardModel=Sequential()
forwardModel.add(Dense(500,input_shape=(4,),init="uniform",activation="sigmoid"))
forwardModel.add(Dense(500,init="uniform",activation="sigmoid"))
forwardModel.add(Dense(400,init="uniform",activation="sigmoid"))
forwardModel.add(Dense(400,init="uniform",activation="sigmoid"))
forwardModel.add(Dense(300,init="uniform",activation="sigmoid"))
forwardModel.add(Dense(3))

# Training, across multiple GPUs if selected/available
model = forwardModel
if (num_gpus > 1):
    model = multi_gpu_model(forwardModel, gpus=num_gpus)

model.compile(optimizer='adam',loss='mse',metrics=['accuracy']) 
model.fit(input_training_data,output_training_data,validation_split=0.2,batch_size=256, epochs=10)
model.save(model_file)



