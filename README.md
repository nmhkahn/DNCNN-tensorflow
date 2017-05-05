# DNCNN-tensorflow
A TensorFlow implementation of "Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising" paper (https://arxiv.org/abs/1608.03981). However, I just refer basic key ideas such as residual learning scheme and loss function and most of the network architecture is different from original paper and code.

## Requirements
1. Python 3.4, 3.5
2. TensorFlow >= 1.0.0
3. numpy, scipy, scikit-image

## Datasets
Any kinds of images can be used in our datasets, but to fit to this code, there is few things you have to check.

1. Datasets directory structures are as follow.

```
```

Since input pipeline see `csv` file before load images, it is important that `csv` file is placed in the right location (e.g. `datasets/train/train.csv`).

2. `csv` file format is described below.

```
reference_image_path,artifact_image_path
reference_image_path,artifact_image_path
reference_image_path,artifact_image_path
...
```

Note that, it is safe to record paths as a absolute path.

3. Very important. This code currently only support **gray** scale image, so to train in colorful image, you must tweek model code to input 3-channel inputs.
