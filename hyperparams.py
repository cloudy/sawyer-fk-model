# parameters for grid search

numhiddenlayers = [1, 2, 3, 4, 5]
neuronsperlayer = [6, 5, 4, 3, 2]
neurons =  [200*neuronsperlayer, 100*neuronsperlayer, 80*neuronsperlayer, 60*neuronsperlayer, 40*neuronsperlayer, 20*neuronsperlayer, 10*neuronsperlayer]
init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]

param_grid = dict(numhiddenlayers = numhiddenlayers, neurons=neurons, init_mode=init_mode, learn_rate=learn_rate, momentum=momentum)


