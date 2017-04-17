python dncnn/train.py \
    --model=residual \
    --checkpoint_basename=residual \
    --logdir=log_residual \
    --quality=10 \
    --image_size=96 \
    --batch_size=32 \
    --learning_rate=0.002 \
    --max_steps=40000 \
    --save_model_steps=5000 \
