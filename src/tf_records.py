"""
Code to create tf_records.
This code is based on https://keras.io/examples/keras_recipes/creating_tfrecords/
"""

import parser
import time
import os
import tensorflow as tf
from glob import glob

######################### ARGUMENTS ################################
ROOT = "/home/monfre/tf_make_dataset/src"
NUM_SAMPLES = 32 # number of samples per tf_record


######################### HELPER FUNCTIONS #########################
def resize_image(img, height, width):
    return tf.image.resize(img, [height, width])


def image_feature(value):
    """Returns a bytes_list from a string / byte."""
    # TODO: does unit8 conversion affect negatively things?
    uint_value = tf.image.convert_image_dtype(value, dtype=tf.uint8)
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(uint_value).numpy()])
    )


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))


def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def float_feature_list(value):
    """Returns a list of float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_example(example):
    # GAN sample
    feature = {
        'image': image_feature(example),
        'label': image_feature(example),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def parse_tfrecord_fn(example):
    """
    Return an example in the form of : image, image
    """
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, feature_description)
    example["image"] = tf.io.decode_jpeg(example["image"], channels=3)
    example["label"] = tf.io.decode_jpeg(example["image"], channels=3)
    return example

######################### MAIN #####################################
def make_dataset(args):
    start = time.time()
    images_dir = ROOT + f"/datasets/raw/{args.path_user}"
    images_list = sorted(glob(images_dir + "/*")) # list with paths to images
    tfrecords_dir = ROOT + f"/datasets/tfrecords/{args.path_user}"
    num_tfrecords = len(images_list) // NUM_SAMPLES

    if not os.path.exists(tfrecords_dir):
        os.makedirs(tfrecords_dir)

    # Generate data in tfrecords format
    for tfrec_num in range(num_tfrecords):
        img_paths = images_list[(tfrec_num*NUM_SAMPLES) : ((tfrec_num+1)*NUM_SAMPLES)]
        # Write samples on a TFRecord
        with tf.io.TFRecordWriter(
            tfrecords_dir + f"/file_{tfrec_num}.tfrec") as writer:
            for img_path in img_paths:
                image = resize_image(
                    tf.io.decode_jpeg(tf.io.read_file(img_path)),
                    args.height, args.width
                )
                example = create_example(image)
                writer.write(example.SerializeToString())

    print(f"Processing time for {len(images_list)} images: {round (time.time() - start, 1)} s")

if __name__ == "__main__":
    make_dataset(parser.parsed_args())