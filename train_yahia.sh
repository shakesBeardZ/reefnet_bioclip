#!/bin/bash
#SBATCH --mem=96G # memory pool for all cores`
#SBATCH --time 23:30:00 # 144
#SBATCH --nodes=1 # number of nodes
#SBATCH --gres=gpu:a100:4 # number of gpus per node
#SBATCH --cpus-per-gpu=10
#SBATCH --mail-type=ALL # 
#SBATCH --mail-user=yahia.battach@kaust.edu.sa
#SBATCH --job-name=ft_bioclip_512
#SBATCH --output=/ibex/project/c2253/ReefVision/bioclip/bioclip/%j-%x.out
#SBATCH --account conf-cvpr2025-2024.11.15-elhosemh

source ~/.bashrc
conda init bash
conda activate /ibex/project/c2253/Xiang_Code/mae/pytoenv

export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun --nproc_per_node 4 -m src.training.main \
  --train-data './data/train/shard-{000000..000159}.tar' \
  --val-data './data/val/shard-{000000..000031}.tar' \
  --root_dir '/ibex/project/c2253/CoralNet_Images/' \
  --csv_file 'annotations_with_aphiaid_and_taxonomy.csv' \
  --patch_type '224' \
  --dataset-type 'auto' \
  --text_type 'taxon' \
  --warmup 1000 \
  --batch-size 256 \
  --image-mean 0.40574995 0.43523004 0.36079923 \
  --image-std 0.13461073 0.14879796 0.13379746 \
  --accum-freq 1 \
  --epochs 100 \
  --workers 10 \
  --model ViT-B-16 \
  --lr 1e-5 \
  --log-every-n-steps 1 \
  --dataset-resampled \
  --local-loss \
  --report-to wandb \
  --gather-with-grad \
  --grad-checkpointing \
  --logs-dir '/ibex/project/c2253/ReefVision/bioclip/bioclip/logs_256/'

