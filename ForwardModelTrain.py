#!/usr/bin/env python

# Training the forward kinematic model of the Sawyer Robot
import glob
import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
from keras.layers import Lambda, concatenate
from keras import Model
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dense
from keras import optimizers
from keras.utils import multi_gpu_model

# Select number of gpus, will only use what is available if requesting more than available
num_gpus = len(device_lib.list_local_devices()) - 1 
if len(sys.argv) > 2:
    num_gpus = min(int(sys.argv[1]), num_gpus)

training_files= glob.glob('data/*.txt') #'data/ufmdata_1.txt'
model_file='models/forwardmodel'

def main():
    for training_file in training_files:
        print("Initializing training for: ", training_file)

        # Dataset to learn the forward kinematics of the Sawyer Robot. Position Prediction only
        training_data=np.loadtxt(training_file,delimiter=',')
        input_training_data=np.delete(training_data,[2,4,6,7,8,9],axis=1)
        output_training_data=np.delete(training_data,[0,1,2,3,4,5,6],axis=1)
        
        # Training, across multiple GPUs if selected/available
        pmodel = model = model_builder()
        if (num_gpus > 1):
            pmodel = multi_gpu_model(model, gpus=num_gpus)
        
        pmodel.compile(optimizer='adam',loss='mse',metrics=['accuracy']) 
        pmodel.fit(input_training_data,output_training_data,validation_split=0.2,batch_size=256, epochs=10)
        model.summary()
        with open(model_file + '_' + training_file.split('_')[-1].split('.')[0] + '_summary.txt', 'w') as model_sum_file: 
            model_sum_file.write(str(model.to_json()))
        model.save(model_file + '_' + training_file.split('_')[-1].split('.')[0] + '.h5')

    # Define the Model
def model_builder():
    Model = Sequential()
    Model.add(Dense(500, input_shape=(4,), kernel_initializer="uniform", activation="sigmoid"))
    Model.add(Dense(500, kernel_initializer="uniform", activation="sigmoid"))
    Model.add(Dense(400, kernel_initializer="uniform", activation="sigmoid"))
    Model.add(Dense(400, kernel_initializer="uniform", activation="sigmoid"))
    Model.add(Dense(300, kernel_initializer="uniform", activation="sigmoid"))
    Model.add(Dense(3))
    return Model

if __name__ == '__main__':
	main()
