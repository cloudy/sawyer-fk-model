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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
from matplotlib.backends.backend_pdf import PdfPages
plt.rcParams.update({'figure.max_open_warning': 0})

#optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

# TODO: Get batch loading, or threaded loading
# TODO: investigate LR
# Select number of gpus, will only use what is available if requesting more than available
num_gpus = len(device_lib.list_local_devices()) - 1 
if len(sys.argv) > 2:
    num_gpus = min(int(sys.argv[1]), num_gpus)

training_files= sorted(glob.glob('/home/michail/sawyer_fk_learning/4DOF/data/*.txt'))
model_file='models/forwardmodel'

def main():
    for training_file in training_files:
        print("Initializing training for: ", training_file)
        filebase = model_file + '_' + training_file.split('/')[-1].split('.')[0] 
        # Dataset to learn the forward kinematics of the Sawyer Robot. Position Prediction only
        training_data=np.loadtxt(training_file,delimiter=',')
        input_training_data = training_data[:,:7] 
        output_training_data = training_data[:,7:] # Issues learning YPR, XYZ fine though
        print(input_training_data.shape)
        print(output_training_data.shape)
        # Training, across multiple GPUs if selected/available
        pmodel = model = model_builder()
        if (num_gpus > 1):
            pmodel = multi_gpu_model(model, gpus=num_gpus)
        
        pmodel.compile(optimizer='adam',loss='mse',metrics=['accuracy']) 
        hist = pmodel.fit(input_training_data, output_training_data, validation_split=0.2, batch_size=256*num_gpus, epochs=30)
        model.summary()
        with open( filebase + '_summary.txt', 'w') as model_sum_file: 
            model_sum_file.write(str(model.to_json()))
        model.save( filebase + '.h5')

        save_plots([plot_performance(hist.history, filebase)], filebase + '_plot.pdf')

def model_builder(numhiddenlayers = 4, init_mode = 'uniform', neurons = [500, 400, 400, 300, 300, 200, 200]):
    Model = Sequential()
    Model.add(Dense(neurons[0], input_shape=(7,), kernel_initializer=init_mode, activation="sigmoid"))
    for i in range(0, numhiddenlayers):
        Model.add(Dense(neurons[i + 1], kernel_initializer=init_mode, activation="sigmoid"))
    Model.add(Dense(6))
    return Model

def plot_performance(hist, ptitle):
    print("Generating performance plot...")
    k = list(hist.keys())
    range_epochs = range(0, len(hist[k[0]]))
    fig = plt.figure()
    plt.title("%s, final val_acc: %f, val_loss: %f" % (ptitle.split('/')[-1], hist['val_acc'][-1], hist['val_loss'][-1]))
    plt.xlabel("epoch")
    plt.ylabel("loss/accuracy")
    plt.ylim(0, 1)
    for res in hist.keys():
        plt.plot(range_epochs, hist[res], label=res)
    plt.legend(loc='upper right')
    print("Performance plot generated. ")
    return fig

def save_plots(figs, filename="dataplot.pdf"):
    with PdfPages(filename) as pdf:
        for fig in figs:
            pdf.savefig(fig)
    print("Plots were saved as: %s" % filename)


if __name__ == '__main__':
	main()
