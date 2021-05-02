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

def print_usage():
    print("Usage: <in datasets> <batch size>")

if __name__ == "__main__":
    symbol_size = 384
    record_size = symbol_size + 1 + 1 + 1 + 8

    MAX_FD = 500

    if len(sys.argv) != 3:
        print_usage()
        sys.exit(1)



    in_dir, batch_size = sys.argv[1:]
    batch_size = int(batch_size)
    
    dataset_paths = get_files_with_suffix_in_dir(in_dir, ".ds")

    for p in dataset_paths:
        assert( "batch-{}".format(batch_size) in p )

    dataset_paths = dataset_paths[:20]
    print(dataset_paths)
    # input("Press enter to continue")

    dataset = tf.data.Dataset.from_tensor_slices(dataset_paths)
    dataset = dataset.shuffle(dataset.cardinality(), reshuffle_each_iteration=True)

    dataset = dataset.interleave(
        lambda path: utils.symbol_dataset_from_file(path, batch_size),
        cycle_length=5, 
        block_length=1,
        deterministic=True
    ).unbatch().filter(lambda a,day,c,d,e: day > 0).batch(100)

    # No filtering: Items per second: 278111.05069821107
    # Filtering: Items per second: 55959.64366670536
    utils.speed_test(dataset, batch_size)

    while True:
        first = True
        for e in dataset:
            if first:
                first = False
                print(e[4][0])
            



        

