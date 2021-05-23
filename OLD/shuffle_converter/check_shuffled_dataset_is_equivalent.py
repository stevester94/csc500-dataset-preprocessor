#! /usr/bin/python3
import io
import pprint
import re
import numpy as np
import os
import sys
import multiprocessing as mp
import random
import time
import tensorflow as tf

import utils


def get_unbatched_dataset_cardinality(ds):
    ds = ds.batch(1000)

    total = 0
    for e in ds:
        total += e[0].shape[0]
    
    return total
    

def print_usage():
    print("Usage: <vanilla binary path> <shuffled dataset path>")

if __name__ == "__main__":

    tf.random.set_seed(1337)

    if len(sys.argv) != 3:
        print_usage()
        sys.exit(1)

    vanilla = sys.argv[1]
    shuffled = sys.argv[2]

    # ds_orig = vanilla_binary_file_to_symbol_dataset("../bin/day-1_transmitter-11_transmission-1.bin")
    ds_vanilla  = utils.vanilla_binary_file_to_symbol_dataset(vanilla)
    ds_shuffled = utils.symbol_dataset_from_file(shuffled, batch_size=1)


    print("Ignore the following error message")
    unshuffled_are_equivalent = utils.check_if_symbol_datasets_are_equivalent(ds_vanilla, ds_shuffled)

    ds_vanilla_cardinality = get_unbatched_dataset_cardinality(ds_vanilla)
    ds_vanilla_shuffled = ds_vanilla.shuffle(ds_vanilla_cardinality)

    shuffled_are_equivalent = utils.check_if_symbol_datasets_are_equivalent(ds_vanilla_shuffled, ds_shuffled)

    print("########################")
    print("Test Results:")
    print("Unshuffled dataset is NOT equivalent to the shuffled dataset:", not unshuffled_are_equivalent)
    print("Shuffled dataset using same seed IS equivalent to the shuffled dataset:", shuffled_are_equivalent)