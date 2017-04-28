import os
import glob
import argparse
import tensorflow as tf
slim = tf.contrib.slim

import trainer

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir",
                        type=str)
    parser.add_argument("--checkpoint_basename",
                        type=str)
    parser.add_argument("--dataset_dir",
                        type=str,
                        default="flickr")
    parser.add_argument("--model",
                        type=str)

    parser.add_argument("--batch_size",
                        type=int,
                        default=32)
    parser.add_argument("--image_size",
                        type=int,
                        default=96)
    
    parser.add_argument("--learning_rate",
                        type=float,
                        default=0.01)
    parser.add_argument("--max_steps",
                        type=int,
                        default=100000)
    parser.add_argument("--beta1",
                        type=float,
                        default=0.5)
    parser.add_argument("--epsilon",
                        type=float,
                        default=0.0001)

    parser.add_argument("--num_threads",
                        type=int,
                        default=4)
    parser.add_argument("--max_to_keep",
                        type=int,
                        default=100)
    parser.add_argument("--summary_every_n_steps",
                        type=int,
                        default=20)
    parser.add_argument("--save_model_steps",
                        type=int,
                        default=5000)
    parser.add_argument("--min_after_dequeue",
                        type=int,
                        default=1000)
    return parser.parse_args()


def main(args):
    filename = os.path.join(args.dataset_dir, "train/train.csv")

    t = trainer.Trainer(filename, args)
    t.fit()

if __name__ == "__main__":
    args = parse_args()
    main(args)
