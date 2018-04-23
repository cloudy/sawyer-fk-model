#/usr/bin/env python3

# Training the forward kinematic model of the Sawyer Robot
import glob
import sys
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dense
from keras import optimizers
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error

from hyperparams import param_grid

training_files = glob.glob('data/*.txt') #'data/ufmdata_1.txt'
model_file = 'models/forwardmodel'

def main():
    for training_file in training_files:
        print("Initializing training for: ", training_file)

        # Dataset to learn the forward kinematics of the Sawyer Robot. Position Prediction only
        training_data = np.loadtxt(training_file, delimiter = ',')
        X, X_test = split_data(np.delete(training_data, [2, 4, 6, 7, 8, 9], axis = 1))
        y, y_test = split_data(np.delete(training_data, [0, 1, 2, 3, 4, 5, 6], axis = 1))
    
        model = KerasRegressor(build_fn = model_builder, nb_epoch = 10, batch_size = 256, verbose = 0)
        grid = GridSearchCV(estimator = model, param_grid = param_grid, n_jobs = -1)
        kfold = KFold(n_splits=10) 
        results = cross_val_score(model, X, y, cv=kfold)
        print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

        model.fit(X, y)
        prediction = model.predict(X_test)
        mean_squared_error(y_test, prediction)
        
        #model.fit(input_training_data, output_training_data, validation_split = 0.2, batch_size = 1024, epochs = 10)
        #model.summary()
        #with open(model_file + '_' + training_file.split('_')[-1].split('.')[0] + '_summary.txt', 'w') as model_sum_file: 
        #    model_sum_file.write(str(model.to_json()))
        #model.save(model_file + '_' + training_file.split('_')[-1].split('.')[0] + '.h5')

    # Define the Model
def model_builder(numhiddenlayers = 4, init_mode = 'uniform', neurons = [500, 400, 300, 200, 100], learn_rate = 0.01, momentum = 0.0):
    Model = Sequential()
    Model.add(Dense(100, input_shape=(4,), kernel_initializer="uniform", activation="sigmoid"))
    for i in range(0, numhiddenlayers):
        Model.add(Dense(neurons[i], kernel_initializer=init_mode, activation="sigmoid"))
    Model.add(Dense(3))
    Model.compile(optimizer='adam',loss='mse',metrics=['accuracy']) 
    return Model

def split_data(data, validation_split = 0.2):
    assert validation_split >= 0 and validation_split <= 1, "Must be between 0 and 1"
    split_pt = int(len(data)*(1 - validation_split))
    return data[:split_pt], data[split_pt:]

if __name__ == '__main__':
	main()
