import tensorflow as tf
import tensorflow.contrib.slim as slim

import ops

def batch_norm_params(is_training):
    return {
        "decay": 0.9,
        "epsilon": 1e-5,
        "scale": True,
        "updates_collections": None,
        "is_training": is_training
    }


def arg_scope(is_training):
    with slim.arg_scope([slim.conv2d],
        weights_initializer=slim.variance_scaling_initializer(),
        activation_fn=ops.lrelu,
        normalizer_fn=slim.batch_norm,
        normalizer_params=batch_norm_params(is_training),
        stride=1, padding="SAME"):
        with slim.arg_scope([slim.batch_norm],
                            **batch_norm_params(is_training)) as arg_scp:
            return arg_scp


def dncnn_base(inputs, reuse=None, scope=None):
    net = inputs
    with tf.variable_scope(scope or "model", reuse=reuse) as scp:
        end_pts_collection = scp.name+"end_pts"
        with slim.arg_scope([slim.conv2d],
                            outputs_collections=end_pts_collection):
            net = slim.conv2d(net, 64, [3, 3],
                              normalizer_fn=None,
                              normalizer_params=None,
                              scope="preconv1")

            for i in range(10):
                net = slim.conv2d(net, 64, [3, 3], scope="block{}/conv1".format(i+1))
                net = slim.conv2d(net, 64, [3, 3], scope="block{}/conv2".format(i+1))
            
            net = slim.conv2d(net, 64, [3, 3], scope="postconv1")
            net = slim.conv2d(net, 64, [3, 3], scope="postconv2")
            net = slim.conv2d(net, 64, [3, 3], scope="postconv3")
            
            net = slim.conv2d(net, 1, [3, 3],
                              activation_fn=tf.nn.tanh,
                              normalizer_fn=None,
                              normalizer_params=None,
                              scope="logit")

            end_pts = slim.utils.convert_collection_to_dict(end_pts_collection)
            dn = inputs - net

    return dn, net, end_pts


def dncnn_residual(inputs, reuse=None, scope=None):
    net = inputs
    with tf.variable_scope(scope or "model", reuse=reuse) as scp:
        end_pts_collection = scp.name+"end_pts"
        with slim.arg_scope([slim.conv2d],
                            outputs_collections=end_pts_collection):
            net = slim.conv2d(net, 64, [3, 3],
                              normalizer_fn=None,
                              normalizer_params=None,
                              scope="preconv1")

            for i in range(10):
                net = ops.residual_block(net, 64, scope="unit{}".format(i+1))
            net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope="postact")
            
            net = slim.conv2d(net, 64, [3, 3], scope="postconv1")
            net = slim.conv2d(net, 64, [3, 3], scope="postconv2")
            net = slim.conv2d(net, 64, [3, 3], scope="postconv3")
            
            net = slim.conv2d(net, 1, [3, 3],
                              activation_fn=tf.nn.tanh,
                              normalizer_fn=None,
                              normalizer_params=None,
                              scope="logit")

            end_pts = slim.utils.convert_collection_to_dict(end_pts_collection)
            dn = inputs - net

    return dn, net, end_pts



def dncnn_base_skip(inputs, reuse=None, scope=None):
    net = inputs
    with tf.variable_scope(scope or "model", reuse=reuse) as scp:
        end_pts_collection = scp.name+"end_pts"
        with slim.arg_scope([slim.conv2d],
                            outputs_collections=end_pts_collection):
            net = slim.conv2d(net, 64, [3, 3],
                              normalizer_fn=None,
                              normalizer_params=None,
                              scope="preconv1")

            shortcut = tf.identity(net, name="shortcut")

            for i in range(10):
                net = slim.conv2d(net, 64, [3, 3], scope="block{}/conv1".format(i+1))
                net = slim.conv2d(net, 64, [3, 3], scope="block{}/conv2".format(i+1))

            with tf.name_scope("skip-conn"):
                net = net + shortcut
            
            net = slim.conv2d(net, 64, [3, 3], scope="postconv1")
            net = slim.conv2d(net, 64, [3, 3], scope="postconv2")
            net = slim.conv2d(net, 64, [3, 3], scope="postconv3")
            
            net = slim.conv2d(net, 1, [3, 3],
                              activation_fn=tf.nn.tanh,
                              normalizer_fn=None,
                              normalizer_params=None,
                              scope="logit")

            end_pts = slim.utils.convert_collection_to_dict(end_pts_collection)
            dn = inputs - net

    return dn, net, end_pts


def dncnn_residual_skip(inputs, reuse=None, scope=None):
    net = inputs
    with tf.variable_scope(scope or "model", reuse=reuse) as scp:
        end_pts_collection = scp.name+"end_pts"
        with slim.arg_scope([slim.conv2d],
                            outputs_collections=end_pts_collection):
            net = slim.conv2d(net, 64, [3, 3],
                              normalizer_fn=None,
                              normalizer_params=None,
                              scope="preconv1")

            shortcut = tf.identity(net, name="shortcut")

            for i in range(10):
                net = ops.residual_block(net, 64, scope="unit{}".format(i+1))
            net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope="postact")

            with tf.name_scope("skip-conn"):
                net =  net + shortcut
            
            net = slim.conv2d(net, 64, [3, 3], scope="postconv1")
            net = slim.conv2d(net, 64, [3, 3], scope="postconv2")
            net = slim.conv2d(net, 64, [3, 3], scope="postconv3")
            
            net = slim.conv2d(net, 1, [3, 3],
                              activation_fn=tf.nn.tanh,
                              normalizer_fn=None,
                              normalizer_params=None,
                              scope="logit")

            end_pts = slim.utils.convert_collection_to_dict(end_pts_collection)
            dn = inputs - net

    return dn, net, end_pts
