#! /usr/bin/python3
import numpy as np
import sys
import tensorflow as tf

import utils

def get_iterator_cardinality(it):
    total = 0
    for e in it:
        total += 1
    
    return total

def binary_file_symbol_generator(path):
    symbol_size=384

    with open(path, "rb") as f:
        buf = f.read(symbol_size)
        while buf:
            yield np.frombuffer(buf,  dtype=np.float32)
            buf = f.read(symbol_size)
        return
    

def print_usage():
    print("Usage: <vanilla binary path>")

if __name__ == "__main__":

    tf.random.set_seed(1337)

    if len(sys.argv) != 2:
        print_usage()
        sys.exit(1)

    vanilla = sys.argv[1]

    ds = utils.vanilla_binary_file_to_symbol_dataset(vanilla)
    # ds = utils.vanilla_binary_file_to_symbol_dataset(vanilla).skip(1) # This failed, GOOD
    ds = ds.as_numpy_iterator()

    ds_cardinality = get_iterator_cardinality(ds)
    vanilla_cardinality = get_iterator_cardinality(binary_file_symbol_generator(vanilla))

    print("ds_cardinality:", ds_cardinality)
    print("vanilla_cardinality:", vanilla_cardinality)
    
    assert(  ds_cardinality == vanilla_cardinality )

    for orig, new in zip(binary_file_symbol_generator(vanilla), ds):
        # print(orig[::2])
        # print(new[0][0])

        assert( np.array_equal( orig[::2], new[0][0] )  )
        assert( np.array_equal( orig[1::2],  new[0][1] ) )

        # print(orig)
        # print(new[0])

        # sys.exit(1)
    
    print("Test Passed")