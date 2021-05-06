#!/bin/sh
#SBATCH --job-name=train_seg_acd_{job_id}
#SBATCH -o logs/log_{job_id}.txt
#SBATCH --time=167:00:00
#SBATCH --partition={gpu}-long
#SBATCH --gres=gpu:{ngpus}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={memory}GB
#SBATCH -d singleton

# Experiment Details :- {top_details}
# Run Details :- {lower_details}

### CODE STARTS
source /mnt/nfs/work1/miyyer/kalpesh/projects/retrieval-lm/.bashrc

BASE_DIR=.

cd /mnt/nfs/work1/miyyer/kalpesh/projects/PointCloudLearningACD
echo $HOSTNAME

python pretrain_partseg_shapenet.py \
    --seed 2001 \
    --k_shot 5 --batch_size 16 --selfsup --step_size 1  --epoch 9 \
    --ss_path data/ACDv2/ \
    --valid_shape_loss_lmbda {valid_shape_loss_lmbda} \
    --perturb_amount {perturb_amount} \
    --perturb_types {perturb_types} \
    --job_id {job_id}
### CODE ENDS
