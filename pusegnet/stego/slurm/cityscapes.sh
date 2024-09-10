#!/bin/bash -l
#SBATCH --job-name=CitySearch
#SBATCH --partition=mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=4
#SBATCH --time=0-03:00:00
#SBATCH --mail-user=benearnthof@hotmail.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=CitySearch.out

source /dss/dsshome1/lxc01/ru25jan4/miniconda3/bin/activate
conda activate /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/miniconda/envs/whisper

export TORCH_HOME=/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/data/cache_dir
export HF_HOME=/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/data/cache_dir
export WANDB_CACHE_DIR=/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/data/cache_dir

NOW=$( date '+%F' )

for index in '0.07 0.4' '0.11 0.35' '0.06 0.375' '0.06 0.4' '0.05 0.375' '0.1 0.375' '0.09 0.375' '0.08 0.4' '0.08 0.375' '0.05 0.4'; do
    set -- $index
    python -W ignore::UserWarning /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/gitroot/stegorep/stegorep/src/train_segmentation.py \
            run_name=STEGO_DINOv2_city_dim_100_${NOW} \
            only_dino=False \
            dimred_type=null \
            pointwise=False \
            dim=100 \
            correspondence_weight=1.0 \
            dataset_name=cityscapes \
            model_type=dinov2_vitb14_reg \
            max_steps=3500 \
            batch_size=32 \
            crop_type=five \
            neg_inter_weight=0.9058762625226623 \
            pos_inter_weight=0.577453483136995 \
            pos_intra_weight=1 \
            neg_inter_shift=0.31361241889448443 \
            pos_inter_shift=$1 \
            pos_intra_shift=$2
done
