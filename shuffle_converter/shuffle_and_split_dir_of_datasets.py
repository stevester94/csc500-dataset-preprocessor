#! /usr/bin/python3
import numpy as np
import sys
import os
import utils
import io
from utils_2 import get_file_size, symbol_tuple_from_bytes, get_files_with_suffix_in_dir, get_iterator_cardinality
from utils import symbol_dataset_from_file
# Keep a big ol list of our paths and current offset, shuffle them with a seed. Randomly select one, open, read from offset, return the buffer.
# Wrap this in a generator
# Use itertools.islice and tf.convert_to_tensor
# utils.write dataset to file should probably be able to take a generator
def binary_random_chunk_generator(seed, paths, chunk_length, disable_randomization=False):    
    rng = np.random.default_rng(seed)

    # The path, current offset, total size of file
    containers = [[path, 0, get_file_size(path)] for path in paths]

    for c in containers:
        if c[2] % chunk_length != 0:
            raise Exception("Path {} is not evenly divisible by {}".format((c[0], c[2])))
    
    while len(containers) > 0:
        if disable_randomization:
            random_index = 0 # Will always read from the first path until it is exhausted, then move to next
        else:
            random_index = rng.integers(0, len(containers)) #randint is exclusive

        c = containers[random_index]

        with open(c[0], "rb") as f:
            f.seek(c[1])
            buf = f.read(chunk_length)
        c[1] += chunk_length
        if c[1] == c[2]:
            containers.remove(c)
        
        yield buf

def test(ds_path_1, ds_path_2):
    symbol_size = 384
    record_size = symbol_size + 1 + 1 + 1 + 8

    ds = symbol_dataset_from_file(ds_path_1, batch_size=1)
    ds = ds.concatenate(symbol_dataset_from_file(ds_path_2, batch_size=1))

    ds_iter = ds.as_numpy_iterator()
    gen = binary_random_chunk_generator(
        1337,
        [ds_path_1, ds_path_2],
        record_size,
        disable_randomization=True
    )


    print("Getting cardinality...")
    ds_cardinality = get_iterator_cardinality(ds_iter)
    gen_cardinality = get_iterator_cardinality(gen)

    print("ds_cardinality:", ds_cardinality)
    print("gen_cardinality:", gen_cardinality)
    
    assert(  ds_cardinality == gen_cardinality )

    # Rebuild our iterators since we exhausted them in the cardinality test
    ds_iter = ds.as_numpy_iterator()
    gen = binary_random_chunk_generator(
        1337,
        [ds_path_1, ds_path_2],
        record_size,
        disable_randomization=True
    )

    print("Comparing all items")
    for orig, new in zip(ds_iter, gen):
        sym = symbol_tuple_from_bytes(new)

        assert( np.array_equal( orig[0], sym[0] ) )
        assert( np.array_equal( orig[1], sym[1] ) )
        assert( np.array_equal( orig[2], sym[2] ) )
        assert( np.array_equal( orig[3], sym[3] ) )
        assert( np.array_equal( orig[4], sym[4] ) )


    ds_iter = ds.as_numpy_iterator()
    gen = binary_random_chunk_generator(
        1337,
        [ds_path_1, ds_path_2],
        record_size,
        disable_randomization=False
    )
    print("Getting cardinality of randomized set")
    ds_cardinality = get_iterator_cardinality(ds_iter)
    gen_cardinality = get_iterator_cardinality(gen)
    print("ds_cardinality:", ds_cardinality)
    print("gen_cardinality:", gen_cardinality)
    assert(  ds_cardinality == gen_cardinality )

    ds_iter = ds.as_numpy_iterator()
    gen = binary_random_chunk_generator(
        1337,
        [ds_path_1, ds_path_2],
        record_size,
        disable_randomization=False
    )
    test_pass = False
    print("Comparing all items again (this time randomized, so should 'fail' quickly)")

    # If we are truely randomizing this will fail quickly
    for orig, new in zip(ds_iter, gen):
        sym = symbol_tuple_from_bytes(new)

        if not np.array_equal( orig[0], sym[0] ):
            test_pass = True
            break
    if not test_pass:
        raise Exception("Generators were equivalent when they should not be")

    print("Test Passed")
    sys.exit(0)



def print_usage():
    print("Usage: <in dir of datasets> <out dir of shuffled and split datasets>")
    print("       test <path of shuffled dataset file> <path of another shuffled dataset file>")

if __name__ == "__main__":


    if len(sys.argv) != 4 and len(sys.argv) != 3:
        print_usage()
        sys.exit(1)

    if sys.argv[1] == "test":
        test(sys.argv[2], sys.argv[3])
        sys.exit(0)

    in_dir, out_dir = sys.argv[1:]
    
    dataset_paths = get_files_with_suffix_in_dir(in_dir, ".ds")

    gen = binary_random_chunk_generator(
        1337,
        dataset_paths,
        record_size,
        disable_randomization=True
    )

    for b in gen:
        t = symbol_tuple_from_bytes(b)
        print(t[4])
        # print(b)