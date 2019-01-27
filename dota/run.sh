#!/bin/bash
clear 

# parameter settings
CHECKPOINT_DIR="logs/dota"
DATASET_NAME="dota"
MODE="train"             # 1 for training, 0 for testing
JSON_FILE="groundtruth.json"
DUP_FLAG=0

# export OMP_NUM_THREADS=1
# export USE_SIMPLE_THREADED_LEVEL3= 1





if [[ $(hostname) = "uranus" ]]; then
echo "Add conda path"
source $HOME/anaconda3/etc/profile.d/conda.sh

echo "Activate py36"
source $HOME/anaconda3/bin/activate py36

echo "export cuda path"
export LD_LIBRARY_PATH=/home/gxdai/cudnn6/lib64:/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH

CUDA_VISIBLE_DEVICES=$1 py_gxdai $2 
elif [[ $(hostname) = "gxdai-Precision-7920-Tower" ]]; then

echo "Add conda path"
source $HOME/anaconda3/etc/profile.d/conda.sh

echo "Activate py36"
source $HOME/anaconda3/bin/activate py36


CUDA_VISIBLE_DEVICES=$1 python $2 
# CUDA_VISIBLE_DEVICES=$1 python $2 
elif [[ $(hostname) = "dgx-r103-2" ]]; then

echo "Add conda path"
source $HOME/anaconda3/etc/profile.d/conda.sh

echo "Activate py36"
source $HOME/anaconda3/bin/activate py36

# CUDA_VISIBLE_DEVICES="0,1,2,3" py_gxdai $1 --dataset_name $DATASET_NAME \
CUDA_VISIBLE_DEVICES="0" py_gxdai $1 --dataset_name $DATASET_NAME \
                                  --json_file $JSON_FILE \
                                  --checkpoint_dir $CHECKPOINT_DIR \
                                  --mode $MODE



fi
