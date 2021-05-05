#! /usr/bin/python3
import numpy as np
import sys
import os
import utils
import io
import itertools
import tensorflow as tf

from steves_utils import utils

def get_original_path_by_metadata(base_path, day, transmitter, transmission):
    file_name = "day-{day}_transmitter-{transmitter}_transmission-{transmission}.bin".format(
        day=day, transmitter=transmitter, transmission=transmission
    )

    path = os.path.join(base_path, file_name)

    if not os.path.exists(path):
        raise Exception("File {} does not exist".format(path))

    return path

def get_symbol_at_index_from_original_path(path, index):
    symbol_size_bytes = 384

    with open(path, "rb") as f:
        f.seek(symbol_size_bytes*index)

        b = f.read(symbol_size_bytes)

    frequency_domain_IQ = np.frombuffer(b, dtype=np.float32)
    frequency_domain_IQ = frequency_domain_IQ.reshape((2,int(len(frequency_domain_IQ)/2)), order="F")

    return frequency_domain_IQ
    

def print_usage():
    print("Usage: <original dir> <shuffled dir> <batch size>")


if __name__ == "__main__":
    symbol_size = 384
    record_size = symbol_size + 1 + 1 + 1 + 8

    MAX_FD = 500


    if len(sys.argv) != 4:
        print_usage()
        sys.exit(1)
    
    original_dir, shuffled_dir, batch_size = sys.argv[1:]
    batch_size = int(batch_size)

    accessor = utils.shuffled_dataset_accessor(
        path=shuffled_dir,
        record_batch_size=batch_size,
        # desired_batch_size=100
        # train_val_test_split = (1, 0, 0)
    )

    ds = accessor["train_ds"]

    for new_iq, day, transmitter, transmission, symbol_index in ds.unbatch():
        # print(transmitter)
        p = get_original_path_by_metadata(original_dir, day, transmitter, transmission)
        
        original_iq = get_symbol_at_index_from_original_path(p, symbol_index)

        print(p, "@", int(symbol_index.numpy()))
        assert( np.array_equal(original_iq, new_iq.numpy()) )
        
