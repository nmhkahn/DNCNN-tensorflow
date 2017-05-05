import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets

bottleneck = nets.resnet_v2.bottleneck


@slim.add_arg_scope
def lrelu(inputs, leak=0.2, scope="lrelu"):
    """
    Note: Implementation is from
    https://github.com/tensorflow/tensorflow/issues/4079
    """
    with tf.variable_scope(scope):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * inputs + f2 * abs(inputs)


def residual_block(inputs,
                   depth, stride=1,
                   outputs_collections=None,
                   scope=None):
    with tf.variable_scope(scope, "residual_block") as scp:
        shortcut = tf.identity(inputs, name="shortcut")
        preact = slim.batch_norm(inputs,
                                 activation_fn=tf.nn.relu,
                                 scope="preact")

        residual = slim.conv2d(preact, depth, [3, 3],
                               stride=stride, scope="conv1")
        residual = slim.conv2d(residual, depth, [3, 3],
                               stride=stride,
                               normalizer_fn=None,
                               activation_fn=None,
                               scope="conv2")

        output = shortcut + residual

    return output


def read_image_from_filename(filename,
                             batch_size, num_threads=4,
                             output_height=128, output_width=128,
                             min_after_dequeue=5000,
                             use_shuffle_batch=True, scope=None):
    with tf.variable_scope(scope, "image_producer"):
        textReader = tf.TextLineReader()

        csv_path = tf.train.string_input_producer([filename])
        _, csv_content = textReader.read(csv_path)
        artifact_filenames, reference_filenames = tf.decode_csv(
            csv_content, record_defaults=[[""], [""]])

        # when training use_shuffle_batch must be True
        # else (e.g. evaluation) evaluation code runs in single epoch and
        # use tf.train.batch instead tf.train.shuffle_batch
        if use_shuffle_batch:
            num_epochs = None
        else:
            num_epochs = 1

        """
        # this method is from https://stackoverflow.com/q/34340489
        # use tf.train.slice_input_producer instead of string_input_producer
        # and tf.read_file instead of tf.WholeFileReader.read
        input_queue = tf.train.slice_input_producer(
            [artifact_filenames, reference_filenames, labels],
            num_epochs=num_epochs, shuffle=False)

        artifact_data  = tf.read_file(input_queue[0])
        reference_data = tf.read_file(input_queue[1])
        label_data     = tf.read_file(input_queue[2])
        """
        artifact_data  = tf.read_file(artifact_filenames)
        reference_data = tf.read_file(reference_filenames)
        artifact_im  = tf.image.decode_jpeg(artifact_data, channels=1)
        reference_im = tf.image.decode_jpeg(reference_data, channels=1)

        # concat all images in channel axis to randomly crop together
        concated_im = tf.concat([artifact_im, reference_im], axis=2)
        if use_shuffle_batch:
            concated_im = tf.random_crop(concated_im,
                                         [output_height, output_width, 1+1])
        elif output_height > 0 and output_width > 0 and not use_shuffle_batch:
            concated_im = tf.image.resize_image_with_crop_or_pad(concated_im,
                                                                 output_height,
                                                                 output_width)

        if use_shuffle_batch:
            capacity = min_after_dequeue + 10 * batch_size
            im_batch = tf.train.shuffle_batch(
                [concated_im],
                batch_size=batch_size,
                capacity=capacity,
                num_threads=num_threads,
                min_after_dequeue=min_after_dequeue,
                allow_smaller_final_batch=True,
                name="shuffle_batch")
        else:
            im_batch, label_batch = tf.train.batch(
                [concated_im],
                batch_size=batch_size,
                num_threads=num_threads,
                allow_smaller_final_batch=True,
                name="batch")

        # split concatenated data
        artifact_batch, reference_batch = tf.split(im_batch, [1, 1], axis=3)
        artifact_batch  = tf.cast(artifact_batch, tf.float32) / 127.5 - 1.0
        reference_batch = tf.cast(reference_batch, tf.float32) / 127.5 - 1.0

        return artifact_batch, reference_batch
