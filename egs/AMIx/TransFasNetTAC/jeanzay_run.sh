#!/bin/bash
#OAR -q production
#OAR -p cluster='graffiti'
#OAR -l /nodes=1,walltime=48:00
# # File where prompts will be outputted
#OAR -O OUT/oar_job.%jobid%.output
# # Files where errors will be outputted
#OAR -E OUT/oar_job.%jobid%.error

# Exit on error
set -e
set -o pipefail

## waiting for 4349186
# Main storage directory where dataset will be stored
#storage_dir=$(readlink -m ./datasets)
# storage_dir=/Users/ccui/Desktop/
# storage_dir=/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/jcosentino/datasets
# librispeech_dir=$storage_dir/LibriSpeech
# noise_dir=$storage_dir/Nonspeech
# After running the recipe a first time, you can run it from stage 3 directly to train new models.

# Path to the python you'll use for the experiment. Defaults to the current python
# You can run ./utils/prepare_python_env.sh to create a suitable python environment, paste the output here.
# This is to activate a python environment with conda

# if [ -f "/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/miniconda3/etc/profile.d/conda.sh" ]; then
# 	. "/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/miniconda3/etc/profile.d/conda.sh"
# 	CONDA_CHANGEPS1=true conda activate speechbrain
# fi
python_path=python
# Example usage
# ./run.sh --stage 3 --tag my_tag --id 0,1

# General
stage=3 # Controls from which stage to start
tag=""  # Controls the directory name associated to the experiment
# You can ask for several GPUs using id (passed to CUDA_VISIBLE_DEVICES)
id=0,1,2,3,4,5,6,7
eval_use_gpu=1

. utils/parse_options.sh

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
	echo "Parsing dataset to json to speed up subsequent experiments"
	for split in train validation test; do
		$python_path ./local/parse_data.py --in_dir $storage_dir/MC_Libri_${dataset_type}/$split --out_json $dumpdir/${split}.json
	done
fi

# Generate a random ID for the run if no tag is specified
uuid=$($python_path -c 'import uuid, sys; print(str(uuid.uuid4())[:8])')
if [[ -z ${tag} ]]; then
	tag=${uuid}
fi

src_type=align # aligned headset
# data_folder=/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/asteroid_pl150/egs/AMIx/TAC/data
data_folder=$WORK/asteroid_pl150/egs/AMIx/TAC/data
dumpdir=${data_folder}/data_clips_monoSpk_8array1_${src_type}_a

# Training config
epochs=100
# batch_size=4 # graffiti 2chn 4 0.84h/epo
# batch_size=3 # grele 2chn 3 2.34h/epo,
batch_size=2 # grele 3chn 2 3.5h/epo, graffiti 3chn 2 1.05h/ep
num_workers=4
half_lr=yes
early_stop=yes
patience=30
save_top_k=5
# Optim config
optim=adam
lr=0.001
weight_decay=0.
# Dataset option
dataset_type=adhoc
samplerate=16000
n_src=3
max_mics=4
segment=4

# exp_dir=exp/TransFasNet_${src_type}_${n_src}spk_chn${max_mics}_bs${batch_size}_epo${epochs}_${tag}
# exp_dir=exp/AMIs_TransFasNetTAC_align_2spk_chn3_bs2_epo5_ded02c59 # waiting graffiti
exp_dir=exp/tmp
# mkdir -p $exp_dir && echo $uuid >>$exp_dir/run_uuid.txt
echo "Results from the following experiment will be stored in $exp_dir"

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
	echo "TAC/TAC" >$exp_dir/publish_dir/recipe_name.txt
fi

if [[ $stage -le 4 ]]; then
	echo "Stage 4 : Evaluation"
	CUDA_VISIBLE_DEVICES=$id $python_path eval_dm.py --test_json $dumpdir/test.json \
		--use_gpu $eval_use_gpu \
		--exp_dir ${exp_dir} | tee logs/eval_${tag}.log
	cp logs/eval_${tag}.log $exp_dir/eval.log
fi
