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
import os
import utils


def get_unbatched_dataset_cardinality(ds):
    ds = ds.batch(1000)

    total = 0
    for e in ds:
        total += e[0].shape[0]
    
    return total
    

def print_usage():
    print("Usage: <unatched dataset path>")

if __name__ == "__main__":

    tf.random.set_seed(1337)

    if len(sys.argv) != 2:
        print_usage()
        sys.exit(1)

    unbatched = sys.argv[1]

    ds_unbatched = utils.symbol_dataset_from_file(unbatched, batch_size=1)
    ds_batched = ds_unbatched.batch(1000, drop_remainder=True)

    utils.symbol_dataset_to_file(ds_batched, "/tmp/batched.ds")
    ds_recovered = utils.symbol_dataset_from_file("/tmp/batched.ds", batch_size=1000)

    print(ds_recovered.element_spec)

    # utils.check_if_symbol_datasets_are_equivalent(ds_batched, ds_recovered)
    utils.check_if_symbol_datasets_are_equivalent(ds_unbatched, ds_recovered.unbatch())

    os.remove("/tmp/batched.ds")