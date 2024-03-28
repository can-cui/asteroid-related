#!/bin/bash
#OAR -q production
#OAR -p cluster='grele'
#OAR -l /nodes=1,walltime=12:00
# # File where prompts will be outputted
#OAR -O OUT/oar_job.%jobid%.output
# # Files where errors will be outputted
#OAR -E OUT/oar_job.%jobid%.error

# Exit on error
set -e
set -o pipefail

# Main storage directory where dataset will be stored
#storage_dir=$(readlink -m ./datasets)

# After running the recipe a first time, you can run it from stage 3 directly to train new models.

# Path to the python you'll use for the experiment. Defaults to the current python
# You can run ./utils/prepare_python_env.sh to create a suitable python environment, paste the output here.
# This is to activate a python environment with conda

if [ -f "/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/miniconda3/etc/profile.d/conda.sh" ]; then
	. "/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/miniconda3/etc/profile.d/conda.sh"
	CONDA_CHANGEPS1=true conda activate speechbrain
fi
python_path=python
# Example usage
# ./run.sh --stage 3 --tag my_tag --id 0,1

# General
stage=4 # Controls from which stage to start
tag=""  # Controls the directory name associated to the experiment
# You can ask for several GPUs using id (passed to CUDA_VISIBLE_DEVICES)
id=0,1,2,3
eval_use_gpu=1

. utils/parse_options.sh

# dumpdir=data_cutaudio_4headsets_2arrays
# check if gpuRIR installed

# if ! (pip list | grep -F gpuRIR); then
# 	echo 'This recipe requires gpuRIR. Please install gpuRIR.'
# 	exit
# fi

if [[ $stage -le 0 ]]; then
	echo "Stage 0: Downloading required Datasets"

	if ! test -e $librispeech_dir/train-clean-100; then
		echo "Downloading LibriSpeech/train-clean-100 into $storage_dir"
		wget -c --tries=0 --read-timeout=20 http://www.openslr.org/resources/12/train-clean-100.tar.gz -P $storage_dir
		tar -xzf $storage_dir/train-clean-100.tar.gz -C $storage_dir
		rm -rf $storage_dir/train-clean-100.tar.gz
	fi

	if ! test -e $librispeech_dir/dev-clean; then
		echo "Downloading LibriSpeech/dev-clean into $storage_dir"
		wget -c --tries=0 --read-timeout=20 http://www.openslr.org/resources/12/dev-clean.tar.gz -P $storage_dir
		tar -xzf $storage_dir/dev-clean.tar.gz -C $storage_dir
		rm -rf $storage_dir/dev-clean.tar.gz
	fi

	if ! test -e $librispeech_dir/test-clean; then
		echo "Downloading LibriSpeech/test-clean into $storage_dir"
		wget -c --tries=0 --read-timeout=20 http://www.openslr.org/resources/12/test-clean.tar.gz -P $storage_dir
		tar -xzf $storage_dir/test-clean.tar.gz -C $storage_dir
		rm -rf $storage_dir/test-clean.tar.gz
	fi

	if ! test -e $storage_dir/Nonspeech; then
		echo "Downloading Noises into $storage_dir"
		wget -c --tries=0 --read-timeout=20 http://web.cse.ohio-state.edu/pnl/corpus/HuNonspeech/Nonspeech.zip -P $storage_dir
		unzip $storage_dir/Nonspeech.zip -d $storage_dir
		rm -rf $storage_dir/Nonspeech.zip
	fi

fi

if [[ $stage -le 1 ]]; then
	echo "Stage 1: Creating Synthetic Datasets"
	git clone https://github.com/yluo42/TAC ./local/TAC
	cd local/TAC/data
	$python_path create_dataset.py \
		--output-path=$storage_dir \
		--dataset=$dataset_type \
		--libri-path=$librispeech_dir \
		--noise-path=$noise_dir
	cd ../../../
fi

if [[ $stage -le 2 ]]; then
	echo "Locally Copying the dataset"
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

	CUDA_VISIBLE_DEVICES=$id $python_path tools/change_json.py --jsondir $jsondir \
		--target_jsondir $target_jsondir \
		--datadir $datadir \
		--target_datadir $target_datadir

	dumpdir=target_jsondir
fi

# Generate a random ID for the run if no tag is specified
uuid=$($python_path -c 'import uuid, sys; print(str(uuid.uuid4())[:8])')
if [[ -z ${tag} ]]; then
	tag=${uuid}
fi

# src_type=st4k # array
src_type=align                                      # aligned headset
dumpdir=data/data_clips_monoSpk_8array1_${src_type} # directory to put generated json file
# Training config
# full epochs: 500
epochs=10
# batch_size=8 # grue 2chn 8 0.5H/epo, grue 3chn 8 0.75h/epo, grele 2chn 8 1h/epo
batch_size=6 # graffiti 2chn 6 0.07h/epo align, grele 3chn 6 1.5h/epo st4k, grele 3chn 6 1.5h/epo st4k
# batch_size=2 # graffiti 3chn 4 0.07h/epo align, grele 3chn 6 1.5h/epo st4k, grele 3chn 6 1.5h/epo st4k
# batch_size=4 # grele 8chn 2 0.63h/epo align, grele 8chn 2nodes
num_workers=4
half_lr=yes
early_stop=yes
patience=50
save_top_k=5
# Optim config
optim=adam
lr=0.001
weight_decay=0.
# Dataset option
dataset_type=adhoc
samplerate=16000
n_src=1
max_mics=2
segment=4
# exp_dir=exp/tmp
# exp_dir=exp/AMIs_TAC_dereverb_${src_type}_${n_src}spk_chn${max_mics}_bs${batch_size}_${tag} # 4358880
# exp_dir=exp/pretrain_AMIs_TAC_dereverb_${src_type}_${n_src}spk_chn${max_mics}_epo${epochs}_bs${batch_size}_${tag} # 4358880
# mkdir -p $exp_dir && echo $uuid >>$exp_dir/run_uuid.txt
# echo "Results from the following experiment will be stored in $exp_dir"

# # if use pretrained model
if [ -f "$expdir/best_model.pth" ]; then
	echo "Pretrained model exists"
else
	# cp -n exp/train360_rir10_TAC_bs4_2spk_2chn_20ep_9f3f47c3/best_model.pth $expdir/ # 1spk
	echo "No pretrained model"
fi

if [[ $stage -le 3 ]]; then
	echo "Stage 3: Training"
	mkdir -p logs
	CUDA_VISIBLE_DEVICES=$id $python_path train_dm.py --train_json $dumpdir/train.json \
		--spk_dict $dumpdir/spk_dict.json \
		--dev_json $dumpdir/validation.json \
		--sample_rate $samplerate \
		--epochs $epochs \
		--batch_size $batch_size \
		--num_workers $num_workers \
		--half_lr $half_lr \
		--early_stop $early_stop \
		--patience $patience \
		--save_top_k $save_top_k \
		--lr $lr \
		--weight_decay $weight_decay \
		--segment $segment \
		--n_src $n_src \
		--max_mics $max_mics \
		--exp_dir ${exp_dir} | tee logs/train_${tag}.log
	cp logs/train_${tag}.log $exp_dir/train.log

	# Get ready to publish
	mkdir -p $exp_dir/publish_dir
	echo "AMI/TAC" >$exp_dir/publish_dir/recipe_name.txt
fi
exp_dir=exp/DAS_4ch
if [[ $stage -le 4 ]]; then
	echo "Stage 4 : Evaluation"
	CUDA_VISIBLE_DEVICES=$id $python_path eval_DAS.py --test_json $dumpdir/test.json \
		--spk_dict $dumpdir/spk_dict.json \
		--use_gpu $eval_use_gpu \
		--max_mics $max_mics \
		--n_src $n_src \
		--exp_dir ${exp_dir} | tee logs/eval_${tag}.log
	cp logs/eval_${tag}.log $exp_dir/eval.log
fi

# test_path=/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/datasets/AMI/csv/noSil/spk/test_noSil_2spk.csv
# if [[ $stage -le 4 ]]; then
# 	echo "Stage 4 : Evaluation"
# 	CUDA_VISIBLE_DEVICES=$id $python_path eval_real.py --test_json $test_path \
# 		\
# 		--use_gpu $eval_use_gpu \
# 		\
# 		--exp_dir ${exp_dir} | tee logs/eval_${tag}.log
# 	cp logs/eval_${tag}.log $exp_dir/eval.log
# fi

# exp_dir=/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/asteroid/egs/TAC/exp/tmp
# if [[ $stage -le 5 ]]; then
# 	# test_dir="/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/asteroid/egs/kinect-wsj/DeepClustering/AMI"
# 	echo "Stage 5 : Test AMI"
# 	CUDA_VISIBLE_DEVICES=$id $python_path test.py \
# 		--n_src $n_src \
# 		--headset_json mix.json \
# 		--use_gpu $eval_use_gpu \
# 		--exp_dir ${exp_dir}
# fi
