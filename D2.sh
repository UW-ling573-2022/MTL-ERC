#!/bin/sh
source ~/anaconda3/etc/profile.d/conda.sh
which conda
ls  ~/anaconda3/envs/UW-LING-573
conda activate ~/anaconda3/envs/UW-LING-573

# put your command for running word2vec.py here
python src/pipeline.py
