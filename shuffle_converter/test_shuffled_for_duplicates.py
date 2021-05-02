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




def check_for_duplicates(shuffled_dir,batch_size):
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

    dataset = dataset.map(
        lambda iq, day, transmitter_id, transmission_id, symbol_index_in_file: 
            (day, transmitter_id, transmission_id, symbol_index_in_file),
            num_parallel_calls=tf.data.AUTOTUNE
    ).unbatch()


    all_combos = []

    count = 0
    for e in dataset:
        inted = (
            int(e[0].numpy()),
            int(e[1].numpy()),
            int(e[2].numpy()),
            int(e[3].numpy()),
        )
        all_combos.append(inted)

        if count % 10000 == 0:
            print(count)
        count += 1

    assert( len(all_combos) == len(set(all_combos)))

    print("Test Passed")

def print_usage():
    print("Usage: <shuffled dir> <batch size>")

if __name__ == "__main__":
    symbol_size = 384
    record_size = symbol_size + 1 + 1 + 1 + 8

    MAX_FD = 500

    if len(sys.argv) != 3:
        print_usage()
        sys.exit(1)


    shuffled_dir, batch_size = sys.argv[1:]
    batch_size = int(batch_size)
    
    check_for_duplicates(shuffled_dir, batch_size)
