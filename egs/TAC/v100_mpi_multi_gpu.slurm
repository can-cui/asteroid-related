#!/bin/bash
#SBATCH --job-name=mpi_multi_gpu     # Name of job
# Other partitions are usable by activating/uncommenting
# one of the 5 following directives:
##SBATCH -C v100-16g                 # uncomment to target only 16GB V100 GPU
##SBATCH -C v100-32g                 # uncomment to target only 32GB V100 GPU
#SBATCH --partition=gpu_p2          # uncomment for gpu_p2 partition (32GB V100 GPU)
##SBATCH --partition=gpu_p4          # uncomment for gpu_p4 partition (40GB A100 GPU)
##SBATCH -C a100                     # uncomment for gpu_p5 partition (80GB A100 GPU)
# Here, reservation of 3x10=30 CPUs (for 3 tasks) and 3 GPUs (1 GPU per task) on a single node:
#SBATCH --nodes=1                    # number of nodes
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node (= number of GPUs per node)
#SBATCH --gres=gpu:8                 # number of GPUs per node (max 8 with gpu_p2, gpu_p4, gpu_p5)
# The number of CPUs per task must be adapted according to the partition used. Knowing that here
# only one GPU per task is reserved (i.e. 1/4 or 1/8 of the GPUs of the node depending on
# the partition), the ideal is to reserve 1/4 or 1/8 of the CPUs of the node for each task:
##SBATCH --cpus-per-task=10           # number of cores per task (1/4 of the node here)
#SBATCH --cpus-per-task=24           # number of cores per task for gpu_p2 (1/8 of 8-GPUs node)
##SBATCH --cpus-per-task=48           # number of cores per task for gpu_p4 (1/8 of 8-GPUs node)
##SBATCH --cpus-per-task=64           # number of cores per task for gpu_p5 (1/8 of 8-GPUs node)
# /!\ Caution, "multithread" in Slurm vocabulary refers to hyperthreading.
#SBATCH --hint=nomultithread         # hyperthreading deactivated
#SBATCH --time=80:00:00              # maximum execution time requested (HH:MM:SS)
#SBATCH --qos=qos_cpu-t4              # To launch jobs exceeding 20 hours
#SBATCH --qos=qos_gpu-t4              # To launch jobs exceeding 20 hours
#SBATCH --output=OUT/mpi_gpu_multi.%j.out # name of output file
#SBATCH --error=OUT/mpi_gpu_multi.%j.out  # name of error file (here, in common with the output file)
##SBATCH -A rte@a100
#SBATCH -A rte@v100

# Cleans out modules loaded in interactive and inherited by default
module purge

# Uncomment the following module command if you are using the "gpu_p5" partition
# to have access to the modules compatible with this partition.
# module load cpuarch/amd

# Loading modules
module load python/3.9.12
conda activate virEnv_py39

# Echo of launched commands
set -x

# For the "gpu_p5" partition, the code must be compiled with the compatible modules.
# Code execution with binding via bind_gpu.sh : 1 GPU per task
srun /gpfslocalsup/pub/idrtools/bind_gpu.sh ./run.sh
# srun ./run.sh
