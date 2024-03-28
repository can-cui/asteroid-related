#!/bin/sh
#OAR -q production
#OAR -p cluster='grele'
##OAR -l core=1,walltime=5:00:00
#OAR -l nodes=1,walltime=5:00:00
#OAR --array-param-file oar100.txt
#OAR -O OUT/oar_job.%jobid%.output
#OAR -E OUT/oar_job.%jobid%.error
set -xv

if [ -f "/home/ccui/miniconda3/etc/profile.d/conda.sh" ]; then
    . "/home/ccui/miniconda3/etc/profile.d/conda.sh"
    CONDA_CHANGEPS1=true conda activate virEnv_py39
fi
python_path=python

python create_data.py $*
