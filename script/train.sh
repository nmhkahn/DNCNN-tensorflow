python dncnn/train.py \
    --image_size=96 \
    --batch_size=32 \
    --checkpoint_basename=base \
    --learning_rate=0.002 \
    --decay_steps=15000 \
    --decay_ratio=0.5 \
    --max_steps=30000 \
    --logdir=log_base/
