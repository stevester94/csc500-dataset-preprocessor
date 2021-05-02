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


def symbol_tuple_generator_wrapper(gen):
    for i in gen:
        yield symbol_tuple_from_bytes(i)

def test(ds_path_1, ds_path_2):
    symbol_size = 384
    record_size = symbol_size + 1 + 1 + 1 + 8

    ds = symbol_dataset_from_file(ds_path_1, batch_size=1)
    ds = ds.concatenate(symbol_dataset_from_file(ds_path_2, batch_size=1))

    ds_iter = ds.as_numpy_iterator()
    gen = symbol_tuple_generator_wrapper(binary_random_chunk_generator(
        1337,
        [ds_path_1, ds_path_2],
        record_size,
        disable_randomization=True
    ))


    print("Getting cardinality...")
    ds_cardinality = get_iterator_cardinality(ds_iter)
    gen_cardinality = get_iterator_cardinality(gen)

    print("ds_cardinality:", ds_cardinality)
    print("gen_cardinality:", gen_cardinality)
    
    assert(  ds_cardinality == gen_cardinality )

    # Rebuild our iterators since we exhausted them in the cardinality test
    ds_iter = ds.as_numpy_iterator()
    gen = symbol_tuple_generator_wrapper(binary_random_chunk_generator(
        1337,
        [ds_path_1, ds_path_2],
        record_size,
        disable_randomization=True
    ))

    print("Comparing all items")
    for orig, sym in zip(ds_iter, gen):
        assert( np.array_equal( orig[0], sym[0] ) )
        assert( np.array_equal( orig[1], sym[1] ) )
        assert( np.array_equal( orig[2], sym[2] ) )
        assert( np.array_equal( orig[3], sym[3] ) )
        assert( np.array_equal( orig[4], sym[4] ) )

    #############################################
    # Now test the randomization
    #############################################
    ds_iter = ds.as_numpy_iterator()
    gen = symbol_tuple_generator_wrapper(binary_random_chunk_generator(
        1337,
        [ds_path_1, ds_path_2],
        record_size,
        disable_randomization=False
    ))
    print("Getting cardinality of randomized set")
    ds_cardinality = get_iterator_cardinality(ds_iter)
    gen_cardinality = get_iterator_cardinality(gen)
    print("ds_cardinality:", ds_cardinality)
    print("gen_cardinality:", gen_cardinality)
    assert(  ds_cardinality == gen_cardinality )

    ds_iter = ds.as_numpy_iterator()
    gen = symbol_tuple_generator_wrapper(binary_random_chunk_generator(
        1337,
        [ds_path_1, ds_path_2],
        record_size,
        disable_randomization=False
    ))
    test_pass = False
    print("Comparing all items again (this time randomized, so should 'fail' quickly)")

    # If we are truely randomizing this will fail quickly
    for orig, sym in zip(ds_iter, gen):
        if not np.array_equal( orig[0], sym[0] ):
            test_pass = True
            break
    if not test_pass:
        raise Exception("Generators were equivalent when they should not be")

    print("Test Passed")
    sys.exit(0)

def print_usage():
    print("Usage: <in dir of datasets> <out dir of shuffled and split datasets> <out batch size> <max file size in MiB>")
    print("       test <path of shuffled dataset file> <path of another shuffled dataset file>")
    print("")

def batcher(generator, batch_size):
    # def _batcher(g):
    #     for i in g:
    #         yield tf.convert_to_tensor(i)

    # def _batcher(generator, batch_size):
    #     while True:
    #         yield list(itertools.islice(generator, batch_size))

    while True:
        # yield list(itertools.islice(_batcher(generator), batch_size))
        # Items per second: 2870058.8476803065
        # yield list(itertools.islice(generator, batch_size))

        # 35406 Items/second It's the best I got
        l = list(itertools.islice(generator, batch_size))

        if len(l) < batch_size:
            return # We've exhausted the generator, and we don't play with incomplete batches

        frequency_domain_IQ = tf.convert_to_tensor([i[0] for i in l], dtype=np.float32)
        day = tf.convert_to_tensor([i[1] for i in l], dtype=np.uint8)
        transmitter_id = tf.convert_to_tensor([i[2] for i in l], dtype=np.uint8)
        transmission_id = tf.convert_to_tensor([i[3] for i in l], dtype=np.uint8)
        symbol_index_in_file = tf.convert_to_tensor([i[4] for i in l], dtype=np.int64)

        yield (
            frequency_domain_IQ,
            day,
            transmitter_id,
            transmission_id,
            symbol_index_in_file,
        )




BYTES_PER_MEBIBYTE = 1024*1024


if __name__ == "__main__":
    symbol_size = 384
    record_size = symbol_size + 1 + 1 + 1 + 8

    if len(sys.argv) != 5 and len(sys.argv) != 3:
        print_usage()
        sys.exit(1)

    if sys.argv[1] == "test":
        test(sys.argv[2], sys.argv[3])
        sys.exit(0)

    if sys.argv[1] == "help":
        print_usage()
        sys.exit(0)  


    in_dir, out_dir, batch_size, max_file_size_MiB = sys.argv[1:]
    batch_size = int(batch_size)
    max_file_size_Bytes = int(max_file_size_MiB) * BYTES_PER_MEBIBYTE
    
    dataset_paths = get_files_with_suffix_in_dir(in_dir, ".ds")

    # dataset_paths = dataset_paths[:4]

    assert( record_size*batch_size <= max_file_size_Bytes )

    print("Will operate on the following paths")
    print(dataset_paths)
    input("Press Enter to continue...")

    gen = symbol_tuple_generator_wrapper(binary_random_chunk_generator(
        1337,
        dataset_paths,
        record_size,
        disable_randomization=False
    ))
    
    # gen = binary_random_chunk_generator(
    #     1337,
    #     dataset_paths,
    #     record_size,
    #     disable_randomization=False
    # )

    # print(get_iterator_cardinality(gen))

    # sys.exit(1)

    out_file_path_format_str = out_dir + "/shuffled_batch-{batch}_part-{part}.ds"



    # 7000 items/sec
    # ds = tf.data.Dataset.from_generator(
    #     lambda: gen,
    #     output_types= (
    #         tf.float32,
    #         tf.uint8,
    #         tf.uint8,
    #         tf.uint8,
    #         tf.int64,
    #     ),
    #     output_shapes=(
    #         (2,48),
    #         (),
    #         (),
    #         (),
    #         (),
    #     )
    # ).batch(2000)
    # utils.speed_test(ds, 2000)
    

    # utils.speed_test(batcher(gen, batch_size), batch_size)

    current_file_index = 0
    current_file_size = 0
    current_file = open(out_file_path_format_str.format(batch=batch_size, part=current_file_index), "wb")
    for batch in batcher(gen, batch_size):
        # bat = [b[0] for b in batch]
        # print(batch)
        # t = tf.convert_to_tensor(batch)
        b = utils.tensor_to_np_bytes(batch)

        if len(b) != record_size*batch_size:
            raise Exception("Expected {} bytes but got {} in buffer".format(record_size*batch_size, len(b)))

        if current_file_size + len(b) > max_file_size_Bytes:
            current_file.close()
            current_file_index += 1
            current_file_size = 0
            current_file = open(out_file_path_format_str.format(batch=batch_size, part=current_file_index), "wb")
            print("Swapping to", out_file_path_format_str.format(batch=batch_size, part=current_file_index))
        current_file.write(b)
        current_file_size += len(b)