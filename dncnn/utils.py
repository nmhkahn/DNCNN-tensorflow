import os
import sys
import scipy.misc
import skimage.measure
import numpy as np

def load_images_from_paths(paths, image_size):
    ims = np.empty((len(paths), image_size, image_size, 3))

    def center_crop(im,
                    output_height,
                    output_width):
        h, w = im.shape[:2]
        if h < output_height and w < output_width:
            raise ValueError("image is small")

        offset_h = int((h - output_height) / 2)
        offset_w = int((w - output_width) / 2)
        return im[offset_h:offset_h+output_height,
                  offset_w:offset_w+output_width, :]

    for i, path in enumerate(paths):
        im = scipy.misc.imread(path)
        im = center_crop(im, image_size, image_size)
        im = im / 127.5 - 1.0
        ims[i] = im

    return ims


def save_image(im, directory, basename, step):
    path = os.path.join(directory, "{}_{}.png".format(step, basename))
    scipy.misc.imsave(path, im)


def compare_psnr(true_im, pred_im):
    return skimage.measure.compare_psnr(true_im, pred_im)


def compare_ssim(true_im, pred_im):
    return skimage.measure.compare_ssim(true_im, pred_im)


def flush_stdout():
    sys.stdout.write("\x1b[1A")
    sys.stdout.write("\x1b[2K")
