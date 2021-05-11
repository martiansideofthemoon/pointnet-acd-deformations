#!/bin/sh
#SBATCH --job-name=finetune_acd_3
#SBATCH -o logs/log_3.txt
#SBATCH --time=167:00:00
#SBATCH --partition=1080ti-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=45GB
#SBATCH -d singleton

# Experiment Details :- GPT2-medium model for margin generation.
# Run Details :- cpus = 3, gpu = rtx8000, memory = 45, ngpus = 1, perturb_amount = 0.5, perturb_types = drop,scale,rotate, valid_shape_loss_lmbda = 10.0

### CODE STARTS
source /mnt/nfs/work1/miyyer/kalpesh/projects/retrieval-lm/.bashrc

BASE_DIR=.

cd /mnt/nfs/work1/miyyer/kalpesh/projects/PointCloudLearningACD
echo $HOSTNAME

python pretrain_partseg_shapenet.py \
    --rotation_z \
    --seed 1001 \
    --model pointnet2_part_seg_msg \
    --batch_size 16 \
    --step_size 1 \
    --selfsup \
    --retain_overlaps \
    --ss_path data/ACDv2 \
    --modelnet_val \
    --valid_shape_loss_lmbda 10.0 \
    --perturb_amount 0.5 \
    --perturb_types drop,scale,rotate \
    --job_id 3
### CODE ENDS

