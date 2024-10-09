#!/bin/bash

# Exit on error
set -e
set -o pipefail

if [ -f "/home/ccui/miniconda3/etc/profile.d/conda.sh" ]; then
    . "/home/ccui/miniconda3/etc/profile.d/conda.sh"
    CONDA_CHANGEPS1=true conda activate virEnv_py39
fi
python_path=python

dir=/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/datasets/AMI
data_name=clips_synthesis_8array1_align
tmp=/tmp/ccui-runtime-dir
datadir=$dir/$data_name
compress=$dir/$data_name.tar.gz
cp -r $compress $tmp
tar -xzf $tmp/$data_name.tar.gz
target_datadir=$tmp/$data_name

data=data_clips_synthesis_4array1_st4k_a
jsondir=./$data
target_jsondir=/tmp/ccui-runtime-dir/$data
mkdir target_jsondir

CUDA_VISIBLE_DEVICES=$id $python_path change_json.py --jsondir $jsondir \
    --target_jsondir $target_jsondir \
    --datadir $datadir \
    --target_datadir $target_datadir

dumpdir=target_jsondir
