#!/bin/sh
#SBATCH --job-name=finetune_acd_test
#SBATCH -o logs/log_test.txt
#SBATCH --time=167:00:00
#SBATCH --partition=1080ti-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=47GB
#SBATCH -d singleton

# Experiment Details :- {top_details}
# Run Details :- {lower_details}

### CODE STARTS
source /mnt/nfs/work1/miyyer/kalpesh/projects/retrieval-lm/.bashrc

cd /mnt/nfs/work1/miyyer/kalpesh/projects/PointCloudLearningACD
BASE_DIR=.

echo $HOSTNAME

python pretrain_partseg_shapenet.py \
    --seed 2001 \
    --k_shot 5 --batch_size 16 --selfsup --step_size 1  --epoch 9 \
    --ss_path data/ACDv2/ \
    --perturb_amount 0.5 \
    --valid_shape_loss_lmbda 1.0 \
    --perturb_types scale,rotate,drop \
    --job_id test_job_seg
### CODE ENDS
