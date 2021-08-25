import parser
import tensorflow as tf
import numpy as np
from glob import glob
from PIL import image

BASE_FOLDER = "/home/monfre/tf_make_dataset/src"
HEIGHT = WIDTH = 256 # Input images must be squared

# Convert images to bytes list
def _bytes_feature(value):
return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# Instance of TFRecord
tf_record_path = BASE_FOLDER + f"/datasets/processed/{args.path_user}"
writer = tf.python_io.TFRecordWriter(tf_record_path)

# Directory of unprcessed images 
data_dir = BASE_FOLDER + f"/datasets/raw/{args.path_user}"
images_path = glob(data_dir + "/*")
for file in images_path:
    img = PIL.Image.open(file)
    img = np.array(img.resize(HEIGHT, WIDTH))
    feature = {
        'image': _bytes_feature(img), 
        'label': _bytes_feature(img.tostring()) # tostring() alias for tobytes(). Convert numpy to bytes
    }
    
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())

writer.close()