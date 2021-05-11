#!/bin/sh
#SBATCH --job-name=train_seg_acd_20
#SBATCH -o logs/log_20.txt
#SBATCH --time=167:00:00
#SBATCH --partition=1080ti-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=45GB
#SBATCH -d singleton

# Experiment Details :- Training PointNet models for ACD + valid shape classification.
# Run Details :- cpus = 3, downstream_type = fewshotseg, gpu = 1080ti, memory = 45, ngpus = 1, perturb_amount = 0.5, perturb_types = drop,scale,rotate, self_sup_lmbda = 0.0, valid_shape_loss_lmbda = 1.0

### CODE STARTS
source /mnt/nfs/work1/miyyer/kalpesh/projects/retrieval-lm/.bashrc

BASE_DIR=.

cd /mnt/nfs/work1/miyyer/kalpesh/projects/PointCloudLearningACD
echo $HOSTNAME

python train_partseg_shapenet_multigpu.py \
    --seed 2001 \
    --k_shot 5 --batch_size 16 --selfsup --step_size 1  --epoch 9 \
    --ss_path data/ACDv2/ \
    --valid_shape_loss_lmbda 1.0 \
    --self_sup_lmbda 0.0 \
    --perturb_amount 0.5 \
    --perturb_types drop,scale,rotate \
    --job_id 20
### CODE ENDS

