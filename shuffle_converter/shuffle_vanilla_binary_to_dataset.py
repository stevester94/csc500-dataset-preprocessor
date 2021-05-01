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
    print("Usage: <source vanilla binary path> <output shuffled dataset path>")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print_usage()
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2]

    # ds_orig = vanilla_binary_file_to_symbol_dataset("../bin/day-1_transmitter-11_transmission-1.bin")
    ds = utils.vanilla_binary_file_to_symbol_dataset(in_path)
    ds_cardinality = get_unbatched_dataset_cardinality(ds)

    print("Items to process:", ds_cardinality)

    ds = ds.shuffle(ds_cardinality)

    utils.symbol_dataset_to_file(ds, out_path)
    