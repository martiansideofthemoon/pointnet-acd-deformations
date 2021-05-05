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

python $BASE_DIR/pretrain_partseg_shapenet.py \
    --rotation_z \
    --seed 1001 \
    --model pointnet2_part_seg_msg \
    --batch_size 16 \
    --step_size 1 \
    --selfsup \
    --retain_overlaps \
    --ss_path data/ACDv2 \
    --perturb_amount 0.5 \
    --modelnet_val \
    --valid_shape_loss_lmbda 1.0 \
    --perturb_types scale,rotate,drop \
    --job_id test_job
### CODE ENDS
