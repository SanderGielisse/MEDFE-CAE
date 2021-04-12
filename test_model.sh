#CUDA_VISIBLE_DEVICES=0 python test.py \
#--st_root="/media/ultra_ssd/celeba/train_val/val" \
#--de_root="/media/ultra_ssd/celeba/train_val/val" \
#--mask_root="/media/ultra_ssd/irregular_mask/disocclusion_img_mask" \
#--name "pretrained"

CUDA_VISIBLE_DEVICES=0 python test.py \
--st_root="/media/ultra_ssd/celeba/train_val/val" \
--de_root="/media/ultra_ssd/celeba/train_val/val" \
--mask_root="/media/ultra_ssd/irregular_mask/disocclusion_img_mask" \
--name "original"