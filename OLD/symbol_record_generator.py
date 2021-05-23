#! /usr/bin/python3
import subprocess
import sys
from typing import List
import json

import hashlib
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
def build_symbol_example(transmitter_id: int, transmission_id: int, day: int, frequency_domain_IQ_tensor, frame_index: int, symbol_index: int):
    assert(len(frequency_domain_IQ_tensor) % 2 == 0)

    serialized_iq_tensor = tf.io.serialize_tensor(frequency_domain_IQ_tensor, name=None)

    feature = {
        'transmitter_id': _int64_feature(transmitter_id),
        'day': _int64_feature(day),
        'transmission_id': _int64_feature(transmission_id),
        'frequency_domain_IQ': _bytes_feature(serialized_iq_tensor),
        'frame_index': _int64_feature(frame_index),
        'symbol_index': _int64_feature(symbol_index),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


# I have found that instantiating this guy on every function call adds non-trivial overhead,
# so declare him outside of the function
_ofdm_symbol_example_description = {
    'transmitter_id':     tf.io.FixedLenFeature([], tf.int64, default_value=-1),
    'day':                tf.io.FixedLenFeature([], tf.int64, default_value=-1),
    'transmission_id':    tf.io.FixedLenFeature([], tf.int64, default_value=-1),
    'frequency_domain_IQ':    tf.io.FixedLenFeature([], tf.string, default_value=''),
    'frame_index':    tf.io.FixedLenFeature([], tf.int64, default_value=-1),
    'symbol_index':    tf.io.FixedLenFeature([], tf.int64, default_value=-1),
}
def parse_serialized_ofdm_symbol_example(serialized_example):
    parsed_example = tf.io.parse_single_example(serialized_example, _ofdm_symbol_example_description)

    parsed_example["frequency_domain_IQ"] = tf.io.parse_tensor(parsed_example["frequency_domain_IQ"], tf.float32)

    # Note that you can actually do some pretty tricky shit here such as
    #return parsed_example["time_domain_IQ"], parsed_example["device_id"]
    #return parsed_example["device_id"]

    return parsed_example

def write_examples_to_records(examples, record_path):
    with tf.io.TFRecordWriter(record_path) as writer:
        for example in examples:
            writer.write(example.SerializeToString())

def np_array_and_sha512_from_file(file_path):
    with open(file_path, "rb") as f:
        buf = f.read()
        sha512 = hashlib.sha512(buf).hexdigest()

        array = np.frombuffer(buf, dtype=np.single)
        return array, sha512

def get_metadata_by_sha512(sha512):
    # Note, breaks with commands printing binary output
    return json.loads(subprocess.getoutput('./metadata_lookup.bash ' + sha512))


if __name__ == "__main__":
    for file_path in sys.argv[1:]:
        print("Processing:", file_path)
        ar, sha512 = np_array_and_sha512_from_file(file_path)
        metadata = get_metadata_by_sha512(sha512)

        day = int(metadata["day"])
        transmitter_id = int(metadata["transmitter_id"])
        transmission_id = int(metadata["transmission_id"])

        # Sanity checks
        assert(file_path == "bin/day-{day}_transmitter-{transmitter}_transmission-{transmission}.bin".format(day=day, transmitter=transmitter_id, transmission=transmission_id))
        assert (metadata["sha512"] == sha512)

        # We convert the vector into a 2d tensor
        iq_tensor = np.array(ar, dtype=np.single)
        iq_tensor = iq_tensor.reshape((2,int(len(ar)/2)), order="F")
        iq_tensor = tf.convert_to_tensor(iq_tensor)

        symbols = tf.signal.frame(iq_tensor, 48, 48)
        symbols = tf.transpose(symbols, perm=[1,0,2]).numpy()

        symbol_examples = []
        for symbol_index in range(symbols.shape[0]):
            symbol_example = build_symbol_example(
                day=day,
                transmitter_id=transmitter_id,
                transmission_id=transmission_id,
                frequency_domain_IQ_tensor=tf.constant(symbols[symbol_index]),
                frame_index=0,
                symbol_index=symbol_index
            )

            symbol_examples.append(symbol_example)

        record_path = "symbol_tfrecords/day-{day}_transmitter-{transmitter}_transmission-{transmission}.tfrecord".format(day=day, transmitter=transmitter_id, transmission=transmission_id)

        print("Writing to:", record_path)
        write_examples_to_records(symbol_examples, record_path)
        print("Complete:", file_path)
        print("=====================================================================")