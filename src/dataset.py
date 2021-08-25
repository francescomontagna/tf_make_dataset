"""
Create a dataset from a folder of jpeg images.
The code process images, performing resizing.

The dataset is meant for GAN training.
It returns a dataset where each sample is a (img, img) tuple. 
"""

import parser
import tensorflow as tf
from glob import glob

BASE_FOLDER = "/home/monfre/tf_make_dataset/src"

# Not all images have the same resolution. 
# Before choosing, I should plot summary (count) of resolutions. Then decide
# If only one image is low res, of course we discard that one. 

# Input images must be squared
HEIGHT = WIDTH = 256

def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_jpeg(img, channels=3)
    return tf.image.resize(img, [HEIGHT, WIDTH])

def process_path(file_path):
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)

    # Second img is for label, remove if not necessary
    return tf.transpose(img, [2, 0, 1]), tf.transpose(img, [2, 0, 1])

# TODO: customize, this is copy and pasted from https://www.tensorflow.org/tutorials/load_data/images
def configure_for_performance(ds):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds

def make_dataset(args):
    data_dir = BASE_FOLDER + f"/datasets/raw/{args.path_user}"
    img_count = len(glob(data_dir + "/*"))
    
    list_ds = tf.data.Dataset.list_files(data_dir + '/[1-9]*.jpg', shuffle=False)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    if args.verbose:
        for image, label in train_ds.take(1):
            print("Image shape: ", image.numpy().shape)
            print("Label: ", label.numpy().shape)

    # Save processed dataset
    output_path = BASE_FOLDER + f"/datasets/processed/{args.path_user}"
    tf.data.experimental.save(train_ds, output_path)  

    # To load the dataset
    if args.verbose:
        new_dataset = tf.data.experimental.load(path=output_path)
        for elem in new_dataset:
            print(elem)
            break

if __name__ == "__main__":
    args = parser.parsed_args()
    HEIGHT, WIDTH = args.height, args.width
    make_dataset(args)

# uso 11.3, potrebbe essere la sola presente chissa dove cazzo