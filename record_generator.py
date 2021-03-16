#! /usr/bin/python3

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Reshape,Dense,Dropout,Activation,Flatten, Convolution1D, MaxPooling2D, ZeroPadding2D, Permute
import tensorflow.keras.models as models
import tensorflow.keras as keras


####################
# Below are ripped from https://www.tensorflow.org/tutorials/load_data/tfrecord
####################

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

####################
# End ripped portion
####################

# Create a dictionary with features that may be relevant.
def build_ofdm_frame_example(device_id_int64, day_int64, ofdm_symbol_bytes, time_domain_IQ_int64):

    feature = {
        'device_id': _int64_feature(device_id_int64),
        'day': _int64_feature(day_int64),
        'ofdm_symbol': _bytes_feature(ofdm_symbol_bytes),
        'time_domain_IQ': _bytes_feature(time_domain_IQ_int64),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


# I have found that instantiating this guy on every function call adds non-trivial overhead,
# so declare him outside of the function
_ofdm_example_description = {
    'device_id'     : tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'time_domain_IQ': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'ofdm_symbol'   : tf.io.FixedLenFeature([], tf.string, default_value=''),
    'day'           : tf.io.FixedLenFeature([], tf.int64, default_value=0),
}
def parse_serialized_ofdm_frame_example(serialized_example):
    parsed_example = tf.io.parse_single_example(serialized_example, _ofdm_example_description)

    # Note that you can actually do some pretty tricky shit here such as
    #return parsed_example["time_domain_IQ"], parsed_example["device_id"]
    return parsed_example["device_id"]
    #return parsed_example

def write_examples_to_records(examples, record_path):
    with tf.io.TFRecordWriter(record_path) as writer:
        for example in examples:
            writer.write(example.SerializeToString())


if __name__ == "__main__":
    ex1 = build_ofdm_frame_example(1, 1, bytes("123", 'utf-8'), bytes("5678", 'utf-8'))
    ex2 = build_ofdm_frame_example(2, 2, bytes("1337", 'utf-8'), bytes("42", 'utf-8'))

    examples = [ex1, ex2]

    write_examples_to_records(examples, "testing.tfrecords")

    # Now read it back
    raw_dataset = tf.data.TFRecordDataset("testing.tfrecords")
    parsed_dataset = raw_dataset.map(parse_serialized_ofdm_frame_example)

    print(parsed_dataset)

    # for features in parsed_dataset:
    #     print(features)

    c = tf.constant([1,2,3], dtype=tf.float32, shape=[3,])
    inputs  = keras.Input(shape=(3,))
    outputs = tf.keras.layers.multiply([inputs, c])

    model = keras.Model(inputs=inputs, outputs=outputs, name="steves_model")

    # model.summary()
    # print( model( tf.constant([2,2,2], dtype=tf.float32, shape=[3,]) ) )


    # # Keras expects either an X and Y dataset for the model, or a single dataset which yields (x, y)