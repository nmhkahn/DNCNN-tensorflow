python dncnn/train.py \
    --dataset_dir=dataset \
    --model=base_skip \
    --checkpoint_basename=dncnn \
    --logdir=log_dncnn \
    --image_size=96 \
    --batch_size=16 \
    --learning_rate=0.002 \
    --max_steps=100000 \
    --save_model_steps=5000 \
