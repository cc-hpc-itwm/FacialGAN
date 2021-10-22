python main.py --mode train --ds_iter 200000 --batch_size 8 \
               --lambda_adv 1 --lambda_sty 1 --lambda_ds 1 --lambda_cyc 1 --lambda_seg 2\
               --eval_every 50000 \
               --attributes "eyes, lip, nose" \
               --train_img_dir data/celeba_hq/train \
               --val_img_dir data/celeba_hq/val \
               --anno_img_dir data/celeba_hq/anno