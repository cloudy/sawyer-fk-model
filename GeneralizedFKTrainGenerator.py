#/usr/bin/env python3

# Training the forward kinematic model of the Sawyer Robot (All 7DOF)
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
from natsort import natsorted
import utils.dataloaders as dl
import utils.grapher as gp

#optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

# TODO: investigate LR

# Select number of gpus, will only use what is available if requesting more than available
num_gpus = len(device_lib.list_local_devices()) - 1 
if len(sys.argv) > 2:
    num_gpus = min(int(sys.argv[1]), num_gpus)

training_files= natsorted(glob.glob('/data/sawyer_fk_data/7DOF/*.txt'))[20:21]
#training_files= natsorted(glob.glob('/data/cloud/sawyer_fk_data/new_data/sawyer_fk_learning/7DOF/data/*.txt'))[20:]
#training_files= natsorted(glob.glob('/home/michail/sawyer_fk_learning/4DOF/data/*.txt'))
model_file='models/forwardmodel'

DOF = 4
POSOR = 3
EPOCHS = 10
BATCHSIZE = 256 * num_gpus
DROPCOL = None

def main():
    for training_file in training_files:
        print("Initializing training for: ", training_file)
        filebase = model_file + '_' + training_file.split('/')[-1].split('.')[0] 
        
        # Training, across multiple GPUs if selected/available
        pmodel = model = model_builder()
        if (num_gpus > 1):
            pmodel = multi_gpu_model(model, gpus=num_gpus)
        
        pmodel.compile(optimizer='adam',loss='mse',metrics=['accuracy']) 
        model.summary()

        steps_cnt = dl.file_len(training_file)//(BATCHSIZE)//EPOCHS 
        print("steps count: ", steps_cnt)

        hist = pmodel.fit_generator(dl.data_generator(training_file, 
                                                    batchsize=BATCHSIZE, 
                                                    dof=DOF, 
                                                    posor=POSOR, 
                                                    dropcol=DROPCOL), 
                                    steps_per_epoch=steps_cnt, 
                                    epochs=EPOCHS)
        
        with open( filebase + '_summary.txt', 'w') as model_sum_file: 
            model_sum_file.write(str(model.to_json()))
        model.save( filebase + '.h5')
        gp.save_plots([gp.plot_performance(hist.history, filebase)], filebase + '_plot.pdf')

def model_builder(numhiddenlayers=3, init_mode='uniform', neurons=[400, 300, 200, 100, 50, 200, 200], acti='relu'):
    Model = Sequential()
    Model.add(Dense(neurons[0], input_shape=(DOF,), kernel_initializer=init_mode, activation=acti))
    for i in range(0, numhiddenlayers):
        Model.add(Dense(neurons[i + 1], kernel_initializer=init_mode, activation=acti))
    Model.add(Dense(POSOR))
    return Model

if __name__ == '__main__':
	main()
