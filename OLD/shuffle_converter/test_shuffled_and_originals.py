#! /usr/bin/python3
import numpy as np
import sys
import os
import utils
import io
from utils import symbol_dataset_from_file, get_file_size, symbol_tuple_from_bytes, get_files_with_suffix_in_dir, get_iterator_cardinality
import utils
import itertools
import tensorflow as tf



def check_first_of_originals(original_dir, shuffled_dir, symbol_size, record_size, batch_size):
    original_paths = get_files_with_suffix_in_dir(original_dir, ".bin") 
    dataset_paths = get_files_with_suffix_in_dir(shuffled_dir, ".ds")

    for p in dataset_paths:
        assert( "batch-{}".format(batch_size) in p )

    dataset = tf.data.Dataset.from_tensor_slices(dataset_paths)
    dataset = dataset.shuffle(dataset.cardinality(), reshuffle_each_iteration=True)

    dataset = dataset.interleave(
        lambda path: utils.symbol_dataset_from_file(path, batch_size),
        cycle_length=MAX_FD, 
        block_length=1,
        deterministic=True
    )

    dataset = dataset.unbatch()

    first_records = []

    for e in dataset:
        if e[4] == 0:
            first_records.append(e)

    print(first_records)
    
    for e in first_records:
        original_name = "day-{day}_transmitter-{transmitter}_transmission-{transmission}.bin".format(day=e[1], transmitter=e[2], transmission=e[3])

        original_path = [p for p in original_paths if original_name in p][0]

        print(original_path)

        ds = utils.vanilla_binary_file_to_symbol_dataset(original_path)

        first_original = next(ds.as_numpy_iterator())

        assert( np.array_equal(first_original[0], e[0].numpy()) )
    
    print("Test Passed")

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
    
    check_first_of_originals(original_dir, shuffled_dir, symbol_size, record_size, batch_size)