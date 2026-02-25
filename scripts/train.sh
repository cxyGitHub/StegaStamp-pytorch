# 修改 encode_image.sh
bash ./scripts/train_imagenet.sh ./asset/images_noise_1x test_run \
  --num_steps 100 \
  --batch_size 2 \
  --save_interval 50 \
  --log_interval 10 \
  --height 224 \
  --width 224 \
  --secret_size 100