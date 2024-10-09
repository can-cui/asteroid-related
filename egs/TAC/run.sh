# !/bin/bash
#OAR -q production
#OAR -p cluster='grele'
#OAR -l /nodes=1,walltime=167:30
# # File where prompts will be outputted
#OAR -O OUT/oar_job.%jobid%.output
# # Files where errors will be outputted
#OAR -E OUT/oar_job.%jobid%.error

# Exit on error
set -e
set -o pipefail

## waiting for run
# Main storage directory where dataset will be stored
storage_dir=$(readlink -m ./datasets)
librispeech_dir=$storage_dir/LibriSpeech
noise_dir=$storage_dir/Nonspeech
# After running the recipe a first time, you can run it from stage 3 directly to train new models.

# Path to the python you'll use for the experiment. Defaults to the current python
# You can run ./utils/prepare_python_env.sh to create a suitable python environment, paste the output here.
# if [ -f "/home/ccui/miniconda3/etc/profile.d/conda.sh" ]; then
# 	. "/home/ccui/miniconda3/etc/profile.d/conda.sh"
# 	CONDA_CHANGEPS1=true conda activate virEnv_py39
# fi
if [ -f "/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/miniconda3/etc/profile.d/conda.sh" ]; then
	. "/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/miniconda3/etc/profile.d/conda.sh"
	CONDA_CHANGEPS1=true conda activate speechbrain
fi
python_path=python

# waiting grue
# generate multi channel data
# $python_path ./data/create_data.py

# Example usage
# ./run.sh --stage 3 --tag my_tag --id 0,1

# General
stage=3 # Controls from which stage to start
tag=""  # Controls the directory name associated to the experiment
# You can ask for several GPUs using id (passed to CUDA_VISIBLE_DEVICES)
id=0,1,2,3,4,5,6,7
# id=0,1,2,3
eval_use_gpu=1

# Dataset option
dataset_type=adhoc
samplerate=16000

. utils/parse_options.sh

dumpdir=data/$suffix # directory to put generated json file

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

n_src=3
n_ch=3
epochs=14
# batch_size=4 # 4 960 1spk grue/graffiti, 3h/epo, 2H/epo
# batch_size=4 # 4 960 2spk grue/graffiti, 3.5h/epo, 2H/epo
# batch_size=4 # 4 960 2spk grue/graffiti, 3.5h/epo, 2H/epo
# batch_size=4 # 4 960 3spk grue/graffiti, 3.5h/epo, 3H/epo
batch_size=4 # grele 4ch 3spk 168 13epo 2spk 168 15epo
num_workers=4
# expdir=exp/tmp
# expdir=exp/960_1000rir_chunk50_win4_accu10_d64_bs${batch_size}_${n_src}spk_${n_ch}chn_${epochs}ep_${tag} # 4329837
# expdir=exp/360_1000rir_123dereverb_rir1000_chunk50_win4_accu10_d64_bs4_1spk_2chn_15ep_1f09e5e4 #
# expdir=exp/960_1000rir_chunk50_win4_accu10_d64_bs4_3spk_3chn_15ep_664ef84a #
# expdir=exp/360_1000rir_rir1000_chunk50_win4_accu10_d64_bs4_3spk_2chn_30ep_0c535349 # 3 spk
# mkdir -p $expdir && echo $uuid >>$expdir/run_uuid.txt
# echo "Results from the following experiment will be stored in $expdir"

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
	CUDA_VISIBLE_DEVICES=$id $python_path -u train.py --sample_rate $samplerate \
		--n_src ${n_src} \
		--n_ch ${n_ch} \
		--exp_dir ${expdir} \
		--epochs $epochs \
		--batch_size $batch_size \
		--num_workers $num_workers | # --n_src $n_src \
		tee logs/train_${tag}.log
	cp logs/train_${tag}.log $expdir/train.log

	# Get ready to publish
	mkdir -p $expdir/publish_dir
	echo "TAC/TAC" >$expdir/publish_dir/recipe_name.txt
fi

out_dir=MC_Libri_Sen
expdir=exp/360_1000rir_rir1000_chunk50_win4_accu10_d64_bs4_2spk_2chn_30ep_c4e646ab

if [[ $stage -le 4 ]]; then
	echo "Stage 4 : Evaluation"
	CUDA_VISIBLE_DEVICES=$id $python_path eval.py --exp_dir $expdir \
		--n_src ${n_src} \
		--n_ch ${n_ch} \
		--out_dir $out_dir \
		--use_gpu $eval_use_gpu \
		--exp_dir ${expdir} | tee logs/eval_${tag}.log
	cp logs/eval_${tag}.log $expdir/eval.log
fi
