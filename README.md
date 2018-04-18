# Sawyer Forward Kinematics ML Model


This will train a model to predict forward kinematics from data acquired from Sawyer robotic arm.

## Instructions to setup

- ssh to GPU machine: `ssh username@192.168.1.27`
- On the `heracleiadl` machine, clone the repository and cd to it:
- `git clone https://github.com/cloudy/sawyer_fk_model.git && cd sawyer_fk_model`
- Create a virtual environment: `mkvirualenv --python=/usr/bin/python3 YOURENVNAME`
- Install packages with pip: `pip install -r requirements.txt`
- Now train: `python ForwardModelTrain.py`

