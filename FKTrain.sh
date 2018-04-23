#PBS -V

source /setups/setup_cuda-8.0.sh

workon nntest

cd /home/cloud/code/sawyer_fk_model/
#python ForwardModelTrain.py
python FKHyperParmOptimization.py
