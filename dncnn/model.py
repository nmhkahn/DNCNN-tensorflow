import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
bottleneck = nets.resnet_v2.bottleneck

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


def dncnn(inputs, reuse=None, scope=None):
    net = inputs
    with tf.variable_scope(scope or "model", reuse=reuse) as scp:
        end_pts_collection = scp.name+"end_pts"
        with slim.arg_scope([slim.conv2d],
                            outputs_collections=end_pts_collection):
            net = slim.conv2d(net, 64, [3, 3],
                              normalizer_fn=None,
                              normalizer_params=None,
                              scope="conv1")

            net = bottleneck(net, 96, 64, 1, scope="unit1")
            net = bottleneck(net, 96, 64, 1, scope="unit2")
            net = bottleneck(net, 96, 64, 1, scope="unit3")
            net = bottleneck(net, 96, 64, 1, scope="unit4")
            net = bottleneck(net, 96, 64, 1, scope="unit5")

            net = slim.conv2d(net, 128, [3, 3], scope="conv2")
            net = slim.conv2d(net, 1, [3, 3],
                              activation_fn=tf.nn.tanh,
                              normalizer_fn=None,
                              normalizer_params=None,
                              scope="conv3")

            end_pts = slim.utils.convert_collection_to_dict(end_pts_collection)
            dn = inputs - net

    return dn, net, end_pts
