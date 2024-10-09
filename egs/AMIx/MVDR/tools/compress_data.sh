#!/bin/bash
#OAR -q production
# OAR -p cluster='grappe'
#OAR -l /nodes=1,walltime=120
# # File where prompts will be outputted
#OAR -O OUT/oar_job.%jobid%.output
# # Files where errors will be outputted
#OAR -E OUT/oar_job.%jobid%.error

# Exit on error
set -e
set -o pipefail

dir=/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/datasets/AMI
data_name=clips_synthesis_8array1_align
data=$dir/$data_name
compress=$dir/$data_name.tar.gz
tar -czf $compress $data
