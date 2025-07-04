# !/bin/bash
#OAR -q production
#OAR -p cluster='graffiti'
#OAR -l /nodes=1,walltime=48
# # File where prompts will be outputted
#OAR -O OUT/oar_job.%jobid%.output
# # Files where errors will be outputted
#OAR -E OUT/oar_job.%jobid%.error

# Exit on error
set -e
set -o pipefail

if [ -f "/home/ccui/miniconda3/etc/profile.d/conda.sh" ]; then
	. "/home/ccui/miniconda3/etc/profile.d/conda.sh"
	CONDA_CHANGEPS1=true conda activate virEnv_py39
fi

# If you haven't generated LibriMix start from stage 0
# Main storage directory. You'll need disk space to store LibriSpeech, WHAM noises
# and LibriMix. This is about 500 Gb
storage_dir=

# After running the recipe a first time, you can run it from stage 3 directly to train new models.

# Path to the python you'll use for the experiment. Defaults to the current python
# You can run ./utils/prepare_python_env.sh to create a suitable python environment, paste the output here.
python_path=python

# Example usage
# ./run.sh --stage 3 --tag my_tag --task sep_noisy --id 0,1

# General
stage=2 # Controls from which stage to start
tag=""  # Controls the directory name associated to the experiment
# You can ask for several GPUs using id (passed to CUDA_VISIBLE_DEVICES)
id=0,1,2,3
out_dir=librimix # Controls the directory name associated to the evaluation results inside the experiment directory

# Network config
n_blocks=8
n_repeats=3
mask_act=relu
# Training config
epochs=40
batch_size=2
num_workers=4
half_lr=yes
early_stop=yes
# Optim config
optimizer=adam
lr=0.001
weight_decay=0.
# Data config
sample_rate=16000
mode=min
n_src=2
# segment=None
task=sep_clean # one of 'enh_single', 'enh_both', 'sep_clean', 'sep_noisy'

eval_use_gpu=1
# Need to --compute_wer 1 --eval_mode max to be sure the user knows all the metrics
# are for the all mode.
compute_wer=0
eval_mode=

. utils/parse_options.sh

sr_string=$(($sample_rate / 1000))
suffix=wav${sr_string}k/$mode

if [ -z "$eval_mode" ]; then
	eval_mode=$mode
fi

# train_dir=data/$suffix/train-360
# valid_dir=data/$suffix/dev
# test_dir=data/wav${sr_string}k/$eval_mode/test

if [[ $stage -le 0 ]]; then
	echo "Stage 0: Generating Librimix dataset"
	if [ -z "$storage_dir" ]; then
		echo "Need to fill in the storage_dir variable in run.sh to run stage 0. Exiting"
		exit 1
	fi
	. local/generate_librimix.sh --storage_dir $storage_dir --n_src $n_src
fi

if [[ $stage -le 1 ]]; then
	echo "Stage 1: Generating csv files including wav path and duration"
	. local/prepare_data.sh --storage_dir $storage_dir --n_src $n_src
fi

# Generate a random ID for the run if no tag is specified
uuid=$($python_path -c 'import uuid, sys; print(str(uuid.uuid4())[:8])')
if [[ -z ${tag} ]]; then
	tag=${uuid}
fi

name=bs${batch_size}_ep${epochs}_sr${sample_rate}_kernel_size128_${tag}
expdir=exp/convtasnet_${name}
mkdir -p $expdir && echo $uuid >>$expdir/run_uuid.txt

echo "Results from the following experiment will be stored in $expdir"

if [[ $stage -le 2 ]]; then
	echo "Stage 2: Training"
	mkdir -p logs
	CUDA_VISIBLE_DEVICES=$id $python_path train.py --exp_dir $expdir \
		\
		\
		\
		--epochs $epochs \
		--batch_size $batch_size \
		--num_workers $num_workers \
		--half_lr $half_lr \
		--early_stop $early_stop \
		--optimizer $optimizer \
		--lr $lr \
		\
		\
		\
		--sample_rate $sample_rate \
		--n_src $n_src \
		\
		--task $task | tee logs/train_${tag}.log
	cp logs/train_${tag}.log $expdir/train.log

	# Get ready to publish
	mkdir -p $expdir/publish_dir
	echo "librispeechMix/ConvTasNet" >$expdir/publish_dir/recipe_name.txt
fi

if [[ $stage -le 3 ]]; then
	echo "Stage 3 : Evaluation"

	if [[ $compute_wer -eq 1 ]]; then
		if [[ $eval_mode != "max" ]]; then
			echo "Cannot compute WER without max mode. Start again with --stage 2 --compute_wer 1 --eval_mode max"
			exit 1
		fi

		# Install espnet if not instaled
		if ! python -c "import espnet" &>/dev/null; then
			echo 'This recipe requires espnet. Installing requirements.'
			$python_path -m pip install espnet_model_zoo
			$python_path -m pip install jiwer
			$python_path -m pip install tabulate
		fi
	fi

	$python_path eval.py \
		--exp_dir $expdir \
		\
		--out_dir $out_dir \
		--use_gpu $eval_use_gpu \
		--compute_wer $compute_wer \
		--task $task | tee logs/eval_${tag}.log

	cp logs/eval_${tag}.log $expdir/eval.log
fi
