#!/bin/bash
clear 

echo "Add conda path"
source $HOME/anaconda3/etc/profile.d/conda.sh

echo "Activate py36"
source $HOME/anaconda3/bin/activate py36

CUDA_VISIBLE_DEVICES=$1 py_gxdai $2 --dataset_name $DATASET_NAME \
                                  --dup_flag $DUP_FLAG \
                                  --train_json $TRAIN_JSON \
                                  --test_json $TEST_JSON \
                                  --image_root_dir $IMAGE_ROOT_DIR \
                                  --checkpoint_dir $CHECKPOINT_DIR \
                                  --instance_number_threshold $INSTANCE_NUMBER_THRESHOLD \
                                  --mode $MODE
