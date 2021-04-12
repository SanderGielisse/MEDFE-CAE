set -ex

# --n_downsampling == 3 --> 32x32
# --n_downsampling == 4 --> 16x16
# --n_downsampling == 5 --> 8x8
# --n_downsampling == 6 --> 4x4

python train.py --dataroot "/media/ultra_ssd/celeba/train_val/train/" --name "celeba_32" --model pix2pix \
--display_port 8094 --lambda_L1 10 --n_epochs 1 --n_epochs_decay 9 --n_downsampling 3 --batch_size 32 &

python train.py --dataroot "/media/ultra_ssd/celeba/train_val/train/" --name "celeba_8" --model pix2pix \
--display_port 8095 --lambda_L1 10 --n_epochs 1 --n_epochs_decay 9 --n_downsampling 5 --batch_size 32 &
