#! /usr/bin/python3
import numpy as np
import sys
import os
import utils
import io
from steves_utils import utils
import itertools
import tensorflow as tf

def print_usage():
    print("Usage: <in datasets> <batch size>")

if __name__ == "__main__":
    tf.random.set_seed(1337)

    symbol_size = 384
    record_size = symbol_size + 1 + 1 + 1 + 8

    MAX_FD = 500

    if len(sys.argv) != 3:
        print_usage()
        sys.exit(1)

    in_dir, batch_size = sys.argv[1:]
    batch_size = int(batch_size)
    
    accessor = utils.shuffled_dataset_accessor(
        path=in_dir,
        record_batch_size=batch_size,
        # desired_batch_size=20000
    )

    all_ds = accessor["all_ds"]
    train_ds = accessor["train_ds"]
    test_ds = accessor["test_ds"]
    val_ds = accessor["val_ds"]
    total_records = accessor["total_records"]


    # for e in train_ds

    # No filtering: Items per second: 278111.05069821107
    # Filtering: Items per second: 55959.64366670536
    # utils.speed_test(all_ds, batch_size)


    # conv can get through 2 full epochs then shits the bed due to the DS being incomplete
    #
    # 

    while True:
        first = True
        count = 0
        for e in train_ds:
            print(e[4])

    # while True:
    #     first = True
    #     count = 0
    #     for e in train_ds:
    #         count += e[0].shape[0]
    #         if first:
    #             first = False
    #             print(e[4][0])
    #     for e in val_ds:
    #         pass
    #     print("Items:",count)
            



        

