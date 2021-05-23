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


def calc_num_chunks_in_file(path, chunk_size):
    size = utils.get_file_size(path)

    assert( size % chunk_size == 0)
    return size / chunk_size


def test_num_records(original_dir, shuffled_dir, symbol_size, record_size, batch_size):
    dataset_paths = get_files_with_suffix_in_dir(shuffled_dir, ".ds")
    bin_paths = get_files_with_suffix_in_dir(original_dir, ".bin")

    total_original_records = sum([calc_num_chunks_in_file(path, symbol_size) for path in bin_paths])
    total_shuffled_records = sum([calc_num_chunks_in_file(path, record_size) for path in dataset_paths])

    print("total_original_records", total_original_records)
    print("total_shuffled_records", total_shuffled_records)

    assert( total_shuffled_records <= total_original_records and total_shuffled_records >= total_original_records-batch_size)


    print("Test Passed")


def print_usage():
    print("Usage: <original dir> <shuffled dir> <shuffled batch size>")

if __name__ == "__main__":
    symbol_size = 384
    record_size = symbol_size + 1 + 1 + 1 + 8

    MAX_FD = 500

    if len(sys.argv) != 4:
        print_usage()
        sys.exit(1)





    original_dir, shuffled_dir, batch_size = sys.argv[1:]
    batch_size = int(batch_size)
    
    test_num_records(original_dir, shuffled_dir, symbol_size, record_size, batch_size)
