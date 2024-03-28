#!/bin/bash
#OAR -q production
#OAR -p cluster='grappe'
#OAR -l core=1,walltime=120
# # File where prompts will be outputted
#OAR -O OUT/oar_job.%jobid%.output
# # Files where errors will be outputted
#OAR -E OUT/oar_job.%jobid%.error

if [ -f "/home/ccui/miniconda3/etc/profile.d/conda.sh" ]; then
    . "/home/ccui/miniconda3/etc/profile.d/conda.sh"
    CONDA_CHANGEPS1=true conda activate virEnv_py39
fi

python parse_data.py
