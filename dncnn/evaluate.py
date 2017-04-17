import os
import glob
import time
import shutil
import argparse
import scipy.misc
import numpy as np

# suppress debugging log
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
slim = tf.contrib.slim

import ops
import utils
import model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop",
                        dest="loop",
                        action="store_true")
    parser.add_argument("--gpu",
                        dest="gpu",
                        action="store_true")

    parser.add_argument("--sample_dir",
                        type=str,
                        default="sample/")
    parser.add_argument("--checkpoint_dir",
                        type=str,
                        default="log/")
    parser.add_argument("--dataset_dir",
                        type=str,
                        default="flickr")

    parser.add_argument("--quality",
                        type=int,
                        default=20)
    parser.add_argument("--num_threads",
                        type=int,
                        default=1)

    return parser.parse_args()


def build_model(im_height, im_width, config):
    artifact_im = tf.placeholder(tf.float32, [1, im_height, im_width, 1])
    is_training = tf.placeholder(tf.bool, name="is_training")

    if config.gpu:
        device = "/gpu:0"
    else:
        device = "/cpu:0"

    model_name = config.checkpoint_dir.split("_")[-1]
    # TODO: more elegant way? (such as factory pattern)
    if model_name == "base":
        model_fn = model.base
    elif model_name == "residual":
        model_fn = model.residual
    elif model_name == "base-skip":
        model_fn = model.base_skip
    elif model_name == "residual-skip":
        model_fn = model.residual_skip
    else:
        raise NotImplementedError("There is no such {} model"
                                  .format(config.model))

    with tf.device(device):
        with slim.arg_scope(model.arg_scope(is_training)):
            dn, residual, _ = model_fn(artifact_im, scope="generator")

    return {"denoised": dn,
            "residual": residual,
            "artifact_im": artifact_im,
            "is_training": is_training}


def wait_for_new_checkpoint(checkpoint_dir, history):
    while True:
        path = tf.train.latest_checkpoint(checkpoint_dir)
        if not path in history:
            break
        time.sleep(300)

    history.append(path)
    return path

def load_from_checkpoint(path, exclude=None):
    init_fn = slim.assign_from_checkpoint_fn(path,
        slim.get_variables_to_restore(exclude=exclude),
        ignore_missing_vars=True)

    return init_fn, path


def loop_body(ckpt_path, artifact_ims, reference_ims, config):
    print(time.ctime())
    print("Load checkpoint: {}".format(ckpt_path))

    mean_psnr_artifact, mean_psnr_denoised = 0, 0
    mean_ssim_artifact, mean_ssim_denoised = 0, 0
    for i in range(len(artifact_ims)):
        artifact_im, reference_im = artifact_ims[i], reference_ims[i]
        h, w = artifact_im.shape[:2]

        params = build_model(h, w, config)
        init_fn, ckpt_path = load_from_checkpoint(ckpt_path)
        ckpt_step = ckpt_path.split("/")[-1]

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        init_fn(sess)

        artifact_im = artifact_im.reshape(1, h, w, 1)

        denoised_im, residual_im = sess.run([
            params["denoised"], params["residual"]],
            feed_dict={params["artifact_im"]: artifact_im,
                       params["is_training"]: False})

        artifact_im  = ((artifact_im+1) / 2.0).reshape((h, w)).astype(np.float32)
        denoised_im  = ((denoised_im+1) / 2.0).reshape((h, w)).astype(np.float32)
        residual_im  = ((residual_im+1) / 2.0).reshape((h, w)).astype(np.float32)
        reference_im = ((reference_im+1) / 2.0).reshape((h, w)).astype(np.float32)

        mean_psnr_artifact = utils.compare_psnr(reference_im,
            artifact_im)
        mean_psnr_denoised = utils.compare_psnr(reference_im,
            denoised_im)

        mean_ssim_artifact = utils.compare_ssim(reference_im,
            artifact_im)
        mean_ssim_denoised = utils.compare_ssim(reference_im,
            denoised_im)

        utils.save_image(artifact_im, config.sample_dir,
                          "{}_artifact".format(i), ckpt_step)
        utils.save_image(reference_im, config.sample_dir,
                          "{}_reference".format(i), ckpt_step)
        utils.save_image(denoised_im, config.sample_dir,
                          "{}_denoised".format(i), ckpt_step)
        utils.save_image(residual_im, config.sample_dir,
                          "{}_residual".format(i), ckpt_step)

        tf.reset_default_graph()

    print("Average PSNR - artifact:{:.3f}, denoising:{:.3f}"
        .format(mean_psnr_artifact, mean_psnr_denoised))
    print("Average SSIM - artifact:{:.3f}, denoising:{:.3f}"
        .format(mean_ssim_artifact, mean_ssim_denoised))


def loop(artifact_ims, origin_ims, config):
    ckpt_history = list()
    while True:
        # wait until new checkpoint exist
        path = wait_for_new_checkpoint(config.checkpoint_dir, ckpt_history)
        loop_body(path, artifact_ims, origin_ims, config)

        if not config.loop: break


def main(config):
    artifact_paths = glob.glob(
        "{}/test/Q{}/*.jpg".format(config.dataset_dir, config.quality))
    reference_paths = glob.glob(
        "{}/test/gray/*.jpg".format(config.dataset_dir))

    artifact_ims  = np.array([scipy.misc.imread(path) / 127.5 - 1.0 
                              for path in artifact_paths])
    reference_ims = np.array([scipy.misc.imread(path) / 127.5 - 1.0 
                              for path in reference_paths])

    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)

    loop(artifact_ims, reference_ims, config)


if __name__ == "__main__":
    config = parse_args()
    main(config)
