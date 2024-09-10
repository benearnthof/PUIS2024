#!/bin/bash -l
#SBATCH --job-name=PotsSearch
#SBATCH --partition=mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=4
#SBATCH --time=0-23:59:59
#SBATCH --mail-user=benearnthof@hotmail.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=PotsSearch.out

source /dss/dsshome1/lxc01/ru25jan4/miniconda3/bin/activate
conda activate /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/miniconda/envs/whisper

export TORCH_HOME=/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/data/cache_dir
export HF_HOME=/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/data/cache_dir
export WANDB_CACHE_DIR=/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/data/cache_dir

NOW=$( date '+%F' )


for inter in $(seq 0.1 0.1 1); do for intra in $(seq 0.1 0.1 1); do
        python -W ignore::UserWarning /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/gitroot/stegorep/stegorep/src/train_segmentation.py \
            run_name=STEGO_pots_DINOv2_70_${NOW} \
            only_dino=False \
            dimred_type=null \
            pointwise=True \
            dim=70 \
            correspondence_weight=1.0 \
            dataset_name=potsdam \
            model_type=dinov2_vits14_reg \
            max_steps=3500 batch_size=16 \
            crop_type=null \
            neg_inter_weight=0.63 \
            pos_inter_weight=0.25 \
            pos_intra_weight=0.67 \
            neg_inter_shift=0.76 \
            pos_inter_shift=$inter \
            pos_intra_shift=$intra
    done
done
