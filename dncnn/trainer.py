import os
import sys
import time
from operator import mul
from functools import reduce
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

import ops
import utils
import model

class Trainer(object):
    def __init__(self, filename, config):
        self.params = dict()
        self.config = config

        self._prepare_inputs(filename)
        self._build_model()

        self.saver = tf.train.Saver(max_to_keep=config.max_to_keep)
        self.loss_summaries = tf.summary.merge([
            tf.summary.scalar("L2_loss", self.params["L2_loss"])
        ])
        self.summary_writer = tf.summary.FileWriter(config.logdir)

        self.sv = tf.train.Supervisor(
            logdir=config.logdir,
            saver=self.saver,
            summary_op=None,
            summary_writer=self.summary_writer,
            save_model_secs=0,
            checkpoint_basename=config.checkpoint_basename,
            global_step=self.params["global_step"])

        sess_config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(allow_growth=True))
        self.sess = self.sv.prepare_or_wait_for_session(config=sess_config)

    def _prepare_inputs(self, filename):
        config, params = self.config, self.params

        global_step  = tf.Variable(0, trainable=False, name="global_step")
        is_training = tf.placeholder(tf.bool, name="is_training")
        learning_rate = tf.Variable(config.learning_rate, 
                                    trainable=False,
                                    name="learning_rate")

        artifact_im, reference_im = ops.read_image_from_filename(
            filename,
            batch_size=config.batch_size, 
            num_threads=config.num_threads,
            output_height=config.image_size, 
            output_width=config.image_size,
            min_after_dequeue=config.min_after_dequeue,
            use_shuffle_batch=True)

        params["is_training"]   = is_training
        params["global_step"]   = global_step
        params["learning_rate"] = learning_rate

        params["artifact_im"]   = artifact_im
        params["reference_im"]  = reference_im

    def _build_model(self):
        config, params = self.config, self.params
        
        is_training   = params["is_training"]
        learning_rate = params["learning_rate"]
        global_step   = params["global_step"]

        artifact_im   = params["artifact_im"]
        reference_im  = params["reference_im"]
       
        # TODO: more elegant way? (such as factory pattern)
        if config.model == "base":
            model_fn = model.base
        elif config.model == "residual":
            model_fn = model.residual
        elif config.model == "base-skip":
            model_fn = model.base_skip
        elif config.model == "residual-skip":
            model_fn = model.residual_skip
        else:
            raise NotImplementedError("There is no such {} model"
                                      .format(config.model))

        with slim.arg_scope(model.arg_scope(is_training)):
            G_dn, G_residual, end_pts = model_fn(
                artifact_im, scope="generator")

        num_params = 0
        for var in tf.trainable_variables():
            print(var.name, var.get_shape())
            num_params += reduce(mul, var.get_shape().as_list(), 1)

        print("Total parameter: {}".format(num_params))

        with tf.variable_scope("Loss"):
            R_residual = artifact_im - reference_im
            # multiple 1000 to compare L2 loss in GAN
            L2_loss = 1000*tf.losses.mean_squared_error(
                labels=R_residual, predictions=G_residual)

        with tf.variable_scope("Optimizer"):
            optimizer = tf.train.AdamOptimizer(
                learning_rate,
                beta1=config.beta1,
                epsilon=config.epsilon).minimize(L2_loss, global_step)

        params["denoised"]  = G_dn
        params["residual"]  = G_residual
        params["L2_loss"]   = L2_loss
        params["optimizer"] = optimizer

    def fit(self):
        config, params = self.config, self.params

        # start training from previous global_step
        start_step = self.sess.run(params["global_step"])
        if not start_step == 0:
            print("Start training from previous {} steps"
                  .format(start_step))

        for step in range(start_step, config.max_steps):
            t1 = time.time()
            self.sess.run(params["optimizer"],
                          feed_dict={params["is_training"]: True})
            t2 = time.time()

            if step % config.summary_every_n_steps == 0:
                summary_feed_dict = {self.params["is_training"]: False}
                self.make_summary(summary_feed_dict, step)

                eta = (t2-t1)*(config.max_steps-step+1)
                print("Finished {}/{} steps, ETA:{:.2f} seconds".format(step,
                    config.max_steps, eta))
                utils.flush_stdout()

            if step % config.save_model_steps == 0:
                self.saver.save(self.sess, os.path.join(config.logdir,
                    "{}-{}".format(config.checkpoint_basename, step)))

        self.saver.save(self.sess, os.path.join(config.logdir,
            "{}-{}".format(config.checkpoint_basename, config.max_steps)))

    def make_summary(self, feed_dict, step):
        summary = self.sess.run(self.loss_summaries, feed_dict=feed_dict)
        self.sv.summary_computed(self.sess, summary, step)
