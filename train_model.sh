# CUDA_VISIBLE_DEVICES=0 python train.py \
# --st_root="/media/ultra_ssd/celeba/train_val/train" \
# --de_root="/media/ultra_ssd/celeba/train_val/train" \
# --mask_root="/media/ultra_ssd/irregular_mask/disocclusion_img_mask" \
# --nThreads=8 --num_workers=8 \
# --display_freq 500 --print_freq 100 \
# --niter 3 --niter_decay 3 --name "l1high" --high_dim_l1

CUDA_VISIBLE_DEVICES=0 python train.py \
--st_root="/media/ultra_ssd/celeba/unused_img_align_celeba_smooth" \
--de_root="/media/ultra_ssd/celeba/train_val/train" \
--mask_root="/media/ultra_ssd/irregular_mask/disocclusion_img_mask" \
--nThreads=8 --num_workers=8 \
--display_freq 500 --print_freq 100 \
--niter 3 --niter_decay 3 --name "original"