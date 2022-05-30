#!/bin/sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ~/anaconda3/envs/UW-LING-573

# Run Primary Task
python src/pipeline.py --do_train --eval_dataset MELD --output_dir "outputs/D4/primary" --result_dir "results/D4/primary"

# Run Adaptation Task
python src/pipeline.py --do_train --eval_dataset MPDD --output_dir "outputs/D4/adaptation" --result_dir "results/D4/adaptation"