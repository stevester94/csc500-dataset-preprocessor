#! /usr/bin/python3
import numpy as np
import sys
import os
import utils
import io

def get_iterator_cardinality(it):
    total = 0
    for e in it:
        total += 1
    
    return total

def get_file_size(path):
    size = 0
    with open(path, "rb") as handle:
        handle.seek(0, io.SEEK_END)
        size = handle.tell()
    
    return size


def symbol_tuple_from_bytes(bytes):
    symbol_size = 384
    record_size = symbol_size + 1 + 1 + 1 + 8

    assert( len(bytes) == record_size )

    # frequency_domain_IQ,  tf.float32 (2,48)
    # day,                  tf.uint8 ()
    # transmitter_id,       tf.uint8 ()
    # transmission_id,      tf.uint8 ()
    # symbol_index_in_file, tf.int64 ()

    frequency_domain_IQ = np.frombuffer(bytes[:symbol_size], dtype=np.float32)
    # frequency_domain_IQ = frequency_domain_IQ.reshape((2,int(len(frequency_domain_IQ)/2)), order="F")
    frequency_domain_IQ = frequency_domain_IQ.reshape((2,int(len(frequency_domain_IQ)/2)))

    day                  = np.frombuffer(bytes[384:385],  dtype=np.uint8)[0]
    transmitter_id       = np.frombuffer(bytes[385:386],  dtype=np.uint8)[0]
    transmission_id      = np.frombuffer(bytes[386:387],  dtype=np.uint8)[0]
    symbol_index_in_file = np.frombuffer(bytes[387:], dtype=np.int64)[0]

    return (
        frequency_domain_IQ,
        day,
        transmitter_id,
        transmission_id,
        symbol_index_in_file,
    )

    # tf.strings.reduce_join(tf.slice(x, [0],                [symbol_size*batch_size])),
    # tf.strings.reduce_join(tf.slice(x, [(symbol_size+0)*batch_size], [1*batch_size])), # Yes it's 0 because we are 0 indexed
    # tf.strings.reduce_join(tf.slice(x, [(symbol_size+1)*batch_size], [1*batch_size])),
    # tf.strings.reduce_join(tf.slice(x, [(symbol_size+2)*batch_size], [1*batch_size])),
    # tf.strings.reduce_join(tf.slice(x, [(symbol_size+3)*batch_size], [8*batch_size])),

def get_files_with_suffix_in_dir(path, suffix):
    (_, _, filenames) = next(os.walk(path))
    return [os.path.join(path,f) for f in filenames if f.endswith(suffix)]