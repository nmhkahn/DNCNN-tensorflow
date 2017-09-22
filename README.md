# DNCNN-tensorflow
A TensorFlow (tf-slim) implementation of "Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising" paper (https://arxiv.org/abs/1608.03981). However, I just refer basic key ideas such as residual learning scheme and loss function and most of the network architecture is different from original paper and code.

## Requirements
1. Python 3.4, 3.5
2. TensorFlow >= 1.0.0
3. numpy, scipy, scikit-image

## Datasets
Any kinds of images can be used in our datasets, but to fit to this code, there is few things you have to check.

1. Datasets directory structures are as follow. Since input pipeline see `csv` file before load images, it is important that `csv` file is placed in the right location (e.g. `datasets/train/train.csv`).

```
dataset
├── train
|   ├── train.csv # this file must be in this location
|   └── images... # train.csv record location of images 
├── val
|   ├── val.csv   # this file must be in this location
|   └── images... # train.csv record location of images 
└── test
    ├── test.csv  # this file must be in this location
    └── images... # train.csv record location of images 
```

2. `csv` file format is described below. Note that, it is safe to record paths as a absolute path. And if you have no information about quality factor, let it to zero.

```
reference_image_path,artifact_image_path,quality_factor
reference_image_path,artifact_image_path,quality_factor
reference_image_path,artifact_image_path,quality_factor
...
```

3. Very important. This code currently only support **gray** scale image, so to train in colorful image, you must tweek model code to handle 3-channel inputs.

## Training and evaluation
`script/train.sh` is example of how to train this network.

```shell
python dncnn/train.py \
    --dataset_dir=dataset \        # directory of dataset to train
    --model=base_skip \            # which model to use? (see dncnn/model.py)
    --checkpoint_basename=dncnn \  # basename used in checkpoint filename
    --logdir=log_dncnn \           # name of log directory which contain checkpoints
    --image_size=96 \              
    --batch_size=16 \
    --learning_rate=0.002 \
    --max_steps=100000 \
    --save_model_steps=5000 \      # save model in every 5000 steps
```

Evaluation is tricky. As far as I know, it is hard to implement train/val different behaviour in TensorFlow input pipeline, so I just split code into train and eval. :(

During training, if you want to see sampled images or PSNR, SSIM metrics, run like below

```shell
# in case evaluate loppy, and use GPU
python dncnn/evaluate.py 
    --sample_dir=sample/ \         # directory will have sampled images
    --checkpoint_dir=log_base \    # checkpoint directory (same as logdir in training code)
    --dataset_dir=dataset
    --loop                         # this code automatically loop evaluate code
    --gpu                          # use GPU
    
# evaluate once! in CPU :(
python dncnn/evaluate.py 
    --sample_dir=sample/ \
    --checkpoint_dir=log_base \
    --dataset_dir=dataset    
```

In `dncnn/evaulate.py`, it repeately check new checkpoint is created and if does, run evaluate with new checkpoint file. And note that, evaluation code can input and output only **single image** at once. Because TensorFlow `placeholder` shape is fixed in building graph phase. So to handle vary size images, this evaluation code is running like below

```python
for image in test_images:
    output = build_graph()
    sess   = tf.Session()
    result = sess.run(output, feed_dict={placeholder:image})
    
    tf.reset_default_graph()
```
