# Self-supervised Learning on Point Clouds using Discriminators on Approximate Convex Decompositions

This repository is a fork from [matheusgadelha/PointCloudLearningACD](https://github.com/matheusgadelha/PointCloudLearningACD), implementing our course project for the COMPSCI674 class on Intelligent Visual Computing (the project is done by Kalpesh Krishna, Melnita Manuel Dabre and Chinmay Shirore). Our key idea is train a discriminator to classify between real and fake 3D point clouds, and use those representations for downstream tasks like unsupervised shape classification and few-shot part segmentation. You can find our report in [report.pdf](report.pdf).

## Setup

Please follow the setup in [matheusgadelha/PointCloudLearningACD](https://github.com/matheusgadelha/PointCloudLearningACD). If you have access to the Gypsum UMass Amherst clusters, you can find the preprocessed data in `/mnt/nfs/work1/miyyer/kalpesh/projects/PointCloudLearningACD/data`.

## Pretrained Models

All our pretrained models can be found on Gypsum UMass Amherst clusters at the path `/mnt/nfs/work1/miyyer/kalpesh/projects/PointCloudLearningACD/saved_models`. The corresponding scripts to run the models can be found in `/mnt/nfs/work1/miyyer/kalpesh/projects/PointCloudLearningACD/slurm-schedulers` (also uploaded to Github). The corresponding training logs are in `/mnt/nfs/work1/miyyer/kalpesh/projects/PointCloudLearningACD/logs`. The mapping from model IDs to configuration is shown below,

Part segmentation jobs ---

1. `model_16` --- `CONTRAST`
2. `model_17`, `model_18`, `model_19`, `model_20` --- `PERTURB-DROP`, `PERTURB-SCALE`, `PERTURB-ROTATE`, `PERTURB-ALL`
3. `model_21`, `model_22`, `model_23`, `model_24` --- Interpolation between `CONTRAST` and `PERTURB-ALL`, with `lambda=0.1,1.0,10,100`.

ModelNet40 / binary classification jobs ---

1. `model_4` --- `CONTRAST`
2. `model_26`, `model_27`, `model_28`, `model_29` --- `PERTURB-DROP`, `PERTURB-SCALE`, `PERTURB-ROTATE`, `PERTURB-ALL`
3. `model_30`, `model_31`, `model_32`, `model_33` --- Interpolation between `CONTRAST` and `PERTURB-ALL`, with `lambda=0.1,1.0,10,100`.

## Evaluating Pretrained Models

For the binary classification experiments, we simply looked at validation performances returned as outputs while the model was training `/mnt/nfs/work1/miyyer/kalpesh/projects/PointCloudLearningACD/logs`. For part segmentation, we used the script `test_segmentation.py` (which we customized from the original codebase), running it five times for each model. Here's an example:

```
python test_segmentation.py --seed 2005 --k_shot 5 --batch_size 16 --selfsup --step_size 1  --epoch 9 --ss_path data/ACDv2/ --valid_shape_loss_lmbda 1.0 --self_sup_lmbda 0.0 --perturb_amount 0.5 --perturb_types rotate --job_id 19 --pretrained saved_models/model_19/checkpoints/model_008.pth
```

For ModelNet40 evaluation, we used the script `test_acdfeat_modelnet.py`. Here's an example:

```
python test_acdfeat_modelnet.py --gpu 0 --sqrt --model pointnet2_part_seg_msg --log_dir saved_models/model_27 --cross_val_svm --ckpt model_005.pth
```

## Running New Jobs

Please check `schedule.py` and `hyperparameters_config.py` which will automate the scheduling process for you with custom hyperparameters. See [`slurm-schedulers`](slurm-schedulers) for configurations used in the paper. If you add more hyperparameters, you will need to modify the `*_template.sh` files as well.
