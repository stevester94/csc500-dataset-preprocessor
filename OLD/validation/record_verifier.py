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
def build_transmission_example(transmitter_id: int, transmission_id: int, day: int, time_domain_IQ: List[float], sha512_of_original: str):
    assert(len(time_domain_IQ) % 2 == 0)

    # We convert the vector into a 2d tensor
    iq_tensor = np.array(time_domain_IQ, dtype=np.single)
    iq_tensor = iq_tensor.reshape((2,int(len(time_domain_IQ)/2)), order="F")
    iq_tensor = tf.convert_to_tensor(iq_tensor)
    serialized_iq_tensor = tf.io.serialize_tensor(iq_tensor, name=None)

    feature = {
        'transmitter_id': _int64_feature(transmitter_id),
        'day': _int64_feature(day),
        'transmission_id': _int64_feature(transmission_id),
        'time_domain_IQ': _bytes_feature(serialized_iq_tensor),
        'sha512_of_original': _bytes_feature(sha512_of_original),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


# I have found that instantiating this guy on every function call adds non-trivial overhead,
# so declare him outside of the function
_ofdm_symbol_example_description = {
    'transmitter_id':     tf.io.FixedLenFeature([], tf.int64, default_value=-1),
    'day':                tf.io.FixedLenFeature([], tf.int64, default_value=-1),
    'transmission_id':    tf.io.FixedLenFeature([], tf.int64, default_value=-1),
    'time_domain_IQ':    tf.io.FixedLenFeature([], tf.string, default_value=''),
    'sha512_of_original': tf.io.FixedLenFeature([], tf.string, default_value=''),
}
def parse_serialized_ofdm_symbol_example(serialized_example):
    parsed_example = tf.io.parse_single_example(serialized_example, _ofdm_symbol_example_description)

    parsed_example["time_domain_IQ"] = tf.io.parse_tensor(parsed_example["time_domain_IQ"], tf.float32)

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
    return json.loads(subprocess.getoutput('cd .. && ./metadata_lookup.bash ' + sha512))

def validate_file_naming_by_metadata(metadata, bin_path, tfrecord_path):
    day = int(metadata["day"])
    transmitter_id = int(metadata["transmitter_id"])
    transmission_id = int(metadata["transmission_id"])

    calculated_bin_path = "../bin/day-{day}_transmitter-{transmitter}_transmission-{transmission}.bin".format(day=day, transmitter=transmitter_id, transmission=transmission_id)
    calculated_tfrecord_path = "../tfrecords/day-{day}_transmitter-{transmitter}_transmission-{transmission}.tfrecord".format(day=day, transmitter=transmitter_id, transmission=transmission_id)

    # Sanity checks for file naming
    if bin_path != calculated_bin_path:
        print("Mismatch in real and calculated bin paths")
        print("bin_path: " + bin_path)
        print("calculated_bin_path: " + calculated_bin_path)
        sys.exit(1)
    if tfrecord_path != calculated_tfrecord_path:
        print("Mismatch in real and calculated tfrecord paths")
        print("tfrecord_path: " + tfrecord_path)
        print("calculated_tfrecord_path: " + calculated_tfrecord_path)
        sys.exit(1)

def validate_data(bin_array, metadata, tfrecord_path):
    day = int(metadata["day"])
    transmitter_id = int(metadata["transmitter_id"])
    transmission_id = int(metadata["transmission_id"])

    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    parsed_dataset = raw_dataset.map(parse_serialized_ofdm_symbol_example)

    tf_iq = tf.data.experimental.get_single_element(parsed_dataset)["time_domain_IQ"].numpy()
    
    num_cols = tf_iq.shape[1]

    assert(num_cols*2 == len(bin_array))
    re_flattened_list = np.zeros(len(bin_array))
    for i in range(num_cols):
        re_flattened_list[i*2] = tf_iq[0][i]
        re_flattened_list[i*2 + 1] = tf_iq[1][i]

    assert(np.array_equal(re_flattened_list, bin_array))



# Usage is [<original bin> <tfrecord version>]
if __name__ == "__main__":
    _argv = sys.argv[1:]
    if len(_argv) % 2 != 0:
        print(len(_argv))
        sys.exit(1)

    targets = ()
    for i in range(0, len(_argv), 2):
        targets += ((_argv[i], _argv[i+1]), )

    for bin_path, tfrecord_path in targets:
        print("Processing ", bin_path, " ", tfrecord_path)

        ar, sha512 = np_array_and_sha512_from_file(bin_path)
        metadata = get_metadata_by_sha512(sha512)

        validate_file_naming_by_metadata(metadata, bin_path, tfrecord_path)
        validate_data(ar, metadata, tfrecord_path)