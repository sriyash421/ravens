#!/bin/bash

# python train.py \
#     data.data_dir=place-red-in-green-train-noise0.002 \
#     train.epochs=1000 \
#     model.mlp=true \
#     data.normalize_actions=true \
#     model.action_heads=1 data.debug=true \
#     data.obs_keys=['ee_pose'] \

    # data.obs_keys=['ee_pose','obj_pose','goal_pose'] \

# python train.py \
#     data.data_dir=place-red-in-green-train-noise0.001 \
#     train.epochs=100 \
#     model.mlp=true \
#     data.normalize_actions=true \
#     model.action_heads=5 \
#     data.obs_keys=['ee_pose','obj_pose','goal_pose'] \

# python ../robomimic/robomimic/scripts/train.py \
#     --config ../robomimic/robomimic/exps/diffusion_policy_ravens_state.json \
#     --name ravens_dp_final_place_red_in_green_$(date +%Y%m%d_%H%M%S) \
#     --dataset dataset.hdf5

# python ../robomimic/robomimic/scripts/train.py \
#     --config ../robomimic/robomimic/exps/bc_ravens_state.json \
#     --name ravens_bc_final_place_red_in_green_$(date +%Y%m%d_%H%M%S) \
#     --dataset dataset_fixed.hdf5

# ckpt="/mmfs1/gscratch/socialrl/sriyash/robomimic/robomimic/../bc_transformer_ravens_state_trained_models/ravens_place_red_in_green_20251021_115839/20251021115928/last.pth"
# ckpt="/gscratch/socialrl/sriyash/robomimic/bc_ravens_trained_models/ravens_place_red_in_green_/20251021152956/models/model_epoch_1450.pth"
# ckpt="/gscratch/socialrl/sriyash/robomimic/diffusion_policy_ravens_trained_models/ravens_place_red_in_green_20251021_155845/20251021160116/models/model_epoch_1200.pth"
# ckpt="/gscratch/socialrl/sriyash/robomimic/bc_ravens_trained_models/ravens_place_red_in_green_20251021_183246/20251021183357/models/model_epoch_50.pth"
# ckpt="../robomimic/diffusion_policy_ravens_trained_models/ravens_place_red_in_green_20251021_155845/20251021160116/models/model_epoch_1450.pth"
# ckpt="../robomimic/bc_ravens_trained_models/ravens_place_red_in_green_20251021_183246/20251021183357/models/model_epoch_250.pth "
# ckpt=$1
# python rollout.py \
#     --checkpoint $ckpt \
#     --config ../robomimic/robomimic/exps/dp_ravens_state.json \
#     --num_episodes 20 \
#     --horizon 400 \
#     --video_dir videos/place_red_in_green_$(date +%Y%m%d_%H%M%S)_$2 --continuous $3

sbatch eval_job.sh ../robomimic/diffusion_policy_ravens_trained_models/ravens_images_all_2k_epochs_final/20251030155314/models/model_epoch_200.pth train_longer_ep200 
sbatch eval_job.sh ../robomimic/diffusion_policy_ravens_trained_models/ravens_images_all_noproprio_final/20251030174355/last.pth no_proprio
sbatch eval_job.sh ../robomimic/diffusion_policy_ravens_trained_models/ravens_images_pretrained_final_1gpu_a100_final/20251030154207/last.pth pretrained_ep50
sbatch eval_job.sh ../robomimic/dit_policy_trained_models/ravens_20251030_153158_all_images_dit/20251030153427/models/model_epoch_500.pth dit_all_images
    
# Submitted batch job 30601270
# Submitted batch job 30601271
# Submitted batch job 30601272
# Submitted batch job 30601273

# 30653220  ckpt         train_longer                     sriyash      0:20       g3043                        N/A      12     128G    
# 30653213  ckpt         dit_pretrained_all               sriyash      1:52       g3043                        N/A      12     128G    
# 30653188  ckpt         dit_pretrained_wrist_only        sriyash      2:54       g3042                        N/A      12     128G    