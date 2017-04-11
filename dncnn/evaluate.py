import os
import glob
import time
import shutil
import argparse
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
    parser.add_argument("--batch_size",
                        type=int,
                        default=10)
    parser.add_argument("--image_size",
                        type=int,
                        default=448)
    parser.add_argument("--num_threads",
                        type=int,
                        default=1)

    return parser.parse_args()


def build_model(filenames, config):
    is_training = tf.placeholder(tf.bool, name="is_training")
    artifact_im, reference_im = ops.read_image_from_filenames(
        filenames,
        base_dir=config.dataset_dir, trainval="test",
        quality=config.quality,
        batch_size=config.batch_size, num_threads=config.num_threads,
        output_height=config.image_size, output_width=config.image_size,
        use_shuffle_batch=False)

    with tf.device("/cpu:0"):
        with slim.arg_scope(model.arg_scope(is_training)):
            dn, residual, _ = model.dncnn(artifact_im, scope="dncnn")

    return {"denoised": dn,
            "residual": residual,
            "reference": reference_im,
            "artifact": artifact_im,
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


def sample_loop(filenames, config):
    ckpt_history = list()
    while True:
        # wait until new checkpoint exist
        path = wait_for_new_checkpoint(config.checkpoint_dir, ckpt_history)

        params = build_model(filenames, config)
        init_fn, ckpt_path = load_from_checkpoint(path)
        ckpt_step = ckpt_path.split("/")[-1]

        sv = tf.train.Supervisor(logdir="/tmp/svtmp",
                                 saver=None,
                                 init_fn=init_fn)
        with sv.managed_session() as sess:
            print(time.ctime())
            print("Load checkpoint: {}".format(ckpt_path))

            ims = [[], [], [], []]
            while True:
                try:
                    denoised_im, residual_im, artifact_im, reference_im = sess.run(
                        [params["denoised"], params["residual"],
                         params["artifact"], params["reference"]],
                        feed_dict={params["is_training"]: False})

                    # re-sacle images
                    ims[0].extend((artifact_im+1) / 2.0)
                    ims[1].extend((reference_im+1) / 2.0)
                    ims[2].extend(((denoised_im+1) / 2.0).astype(np.float32))
                    ims[3].extend(((residual_im+1) / 2.0).astype(np.float32))
                except tf.errors.OutOfRangeError:
                    break
            ims = [np.array(element).reshape(
                   (-1, config.image_size, config.image_size))
                   for element in ims]

        # evaluate metric and save samples
        mean_psnr_artifact, mean_psnr_denoised = 0, 0
        for artifact, reference, denoised in zip(ims[0], ims[1], ims[2]):
            mean_psnr_artifact += utils.compare_psnr(reference,
                artifact) / len(ims[1])
            mean_psnr_denoised  += utils.compare_psnr(reference,
                denoised) / len(ims[1])

        print("Average PSNR - artifact:{:.3f}, denoising:{:.3f}"
            .format(mean_psnr_artifact, mean_psnr_denoised))

        utils.save_images(ims[0], config.sample_dir,
                          "artifact", ckpt_step)
        utils.save_images(ims[1], config.sample_dir,
                          "reference", ckpt_step)
        utils.save_images(ims[2], config.sample_dir,
                          "denoised", ckpt_step)
        utils.save_images(ims[3], config.sample_dir,
                          "residual", ckpt_step)

        tf.reset_default_graph()
        shutil.rmtree("/tmp/svtmp")

        if not config.loop:
            break


def main(config):
    im_paths = glob.glob("{}/test/Q{}/*.jpg".format(config.dataset_dir,
                                                   config.quality))[:300]
    im_names = [path.split(".")[0].split("/")[-1] for path in im_paths]

    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)

    sample_loop(im_names, config)


if __name__ == "__main__":
    config = parse_args()
    main(config)
