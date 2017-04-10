python dncnn/train.py \
    --image_size=96 \
    --batch_size=32 \
    --checkpoint_basename=dncnn \
    --learning_rate=0.001 \
    --decay_steps=25000 \
    --decay_ratio=0.5 \
    --max_steps=100000 \
    --logdir=log/
