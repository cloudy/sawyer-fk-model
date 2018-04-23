# Sawyer Forward Kinematics ML Model


This will train a model to predict forward kinematics from data acquired from Sawyer robotic arm.

## Instructions to setup

- ssh to GPU machine: `ssh -XY username@192.168.1.27`
- On the `heracleiadl` machine, clone the repository and cd to it:
- `git clone https://github.com/cloudy/sawyer_fk_model.git && cd sawyer_fk_model`
- Create a virtual environment: `mkvirtualenv --python=/usr/bin/python3 YOURENVNAME`
- Install packages with pip: `pip install -r requirements.txt`
- Now train: `python ForwardModelTrain.py`

## Adding data

If attempting to use data that is not presently on `heracleiadl`.

You can scp your data files over from your current machine.

For example, you have file1.txt stored in /home/username/dir,

and you want to move it to /home/username/dir2/dir4/ on `heracleiadl`.

The scp command to do this is `scp /home/username/dir/file1.txt username@192.168.1.27:/home/username/dir2/dir4`
