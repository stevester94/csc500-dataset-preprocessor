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

def metadata_from_path(path):
    match  = re.search("day-([0-9]+)_transmitter-([0-9]+)_transmission-([0-9]+)", path)
    (day, transmitter_id, transmission_id) = match.groups()

    return {
        "day": int(day),
        "transmitter_id": int(transmitter_id),
        "transmission_id": int(transmission_id)
    }



def vanilla_binary_file_to_symbol_dataset(
    binary_path
):
    symbol_size=384

    metadata = metadata_from_path(binary_path)

    dataset = tf.data.FixedLengthRecordDataset(
        binary_path, record_bytes=symbol_size, header_bytes=None, footer_bytes=None, buffer_size=None,
        compression_type=None, num_parallel_reads=1
    )

    dataset = dataset.enumerate()

    # frequency_domain_IQ,  tf.float32 (2,48)
    # day,                  tf.uint8 ()
    # transmitter_id,       tf.uint8 ()
    # transmission_id,      tf.uint8 ()
    # symbol_index_in_file, tf.int64 ()

    dataset = dataset.map(
        lambda index,frequency_domain_IQ: (
            tf.io.decode_raw(frequency_domain_IQ, tf.float32),
            tf.constant(metadata["day"], dtype=tf.uint8),
            tf.constant(metadata["transmitter_id"], dtype=tf.uint8),
            tf.constant(metadata["transmission_id"], dtype=tf.uint8),
            tf.cast(index, dtype=tf.int64)
        ),
        num_parallel_calls=10,
        deterministic=True
    )

    dataset = dataset.map(
        lambda frequency_domain_IQ,day,transmitter_id,transmission_id,symbol_index_in_file: (
            tf.reshape(frequency_domain_IQ, (2,48)),
            tf.reshape(day, () ),
            tf.reshape(transmitter_id, () ),
            tf.reshape(transmission_id, () ),
            tf.reshape(symbol_index_in_file, () ),
        ),
        num_parallel_calls=10,
        deterministic=True
    )


    return dataset


def symbol_dataset_to_file(dataset, out_path):
    with open(out_path, "wb") as f:
        for e in dataset:
            frequency_domain_IQ, day, transmitter_id, transmission_id, symbol_index_in_file = e

            members = [frequency_domain_IQ, day, transmitter_id, transmission_id, symbol_index_in_file]

            members_as_numpy = [m.numpy() for m in members]
            members_as_buffer = [m_np.tobytes() for m_np in members_as_numpy]
            
            for buffer in members_as_buffer:
                f.write(buffer)


# If batch_size is 1, then will not batch at all, else will parse as batches.
# NB: Batch size must match the batch setting that the file was created with
def symbol_dataset_from_file(path, batch_size):
    symbol_size=384
    record_size=symbol_size + 1 + 1 + 1 + 8
    
    dataset = tf.data.FixedLengthRecordDataset(
        path, record_bytes=record_size, header_bytes=None, footer_bytes=None, buffer_size=None,
        compression_type=None, num_parallel_reads=1
    )

    # return dataset

    dataset = dataset.map(
        lambda x: tf.strings.bytes_split(
            x, name=None
        )
    )

    # print(dataset.element_spec)
    # sys.exit(1)

    # dataset = dataset.map(
    #     lambda x: tf.io.decode_raw(
    #         x, tf.uint8, little_endian=True, fixed_length=None, name=None
    #     )
    # )

    # dataset = dataset.map(
    #     lambda x: tf.reshape(x, (symbol_size,))
    # )

    # frequency_domain_IQ,  tf.float32 (2,48)
    # day,                  tf.uint8 ()
    # transmitter_id,       tf.uint8 ()
    # transmission_id,      tf.uint8 ()
    # symbol_index_in_file, tf.int64 ()



    # for e in dataset.take(1):
    #     print(tf.rank(e))
    
    # c = tf.constant([1,2,3,4,5,6,7,8])
    # o = tf.slice(c, [0], [1])

    # print(o)


    dataset = dataset.map(
        lambda x: (
            tf.strings.reduce_join(tf.slice(x, [0],                [symbol_size*batch_size])),
            # tf.strings.reduce_join(tf.slice(x, [],                [1*batch_size])),
            tf.strings.reduce_join(tf.slice(x, [(symbol_size+0)*batch_size], [1*batch_size])), # Yes it's 0 because we are 0 indexed
            tf.strings.reduce_join(tf.slice(x, [(symbol_size+1)*batch_size], [1*batch_size])),
            tf.strings.reduce_join(tf.slice(x, [(symbol_size+2)*batch_size], [1*batch_size])),
            tf.strings.reduce_join(tf.slice(x, [(symbol_size+3)*batch_size], [8*batch_size])),
        ),
        num_parallel_calls=10,
        deterministic=True
    )

    # return dataset

    # dataset = dataset.map(
    #     lambda x: (
    #         tf.slice(x, [0],                [symbol_size*batch_size]),
    #         tf.slice(x, [(symbol_size+0)*batch_size], [1*batch_size]), # Yes it's 0 because we are 0 indexed
    #         tf.slice(x, [(symbol_size+1)*batch_size], [1*batch_size]),
    #         tf.slice(x, [(symbol_size+2)*batch_size], [1*batch_size]),
    #         tf.slice(x, [(symbol_size+3)*batch_size], [8*batch_size]),
    #     ),
    #     num_parallel_calls=10,
    #     deterministic=True
    # )


    # dataset = dataset.map(
    #     lambda frequency_domain_IQ, day, transmitter_id, transmission_id, symbol_index_in_file: (
    #         tf.strings.join(frequency_domain_IQ),
    #         tf.strings.join(day),
    #         tf.strings.join(transmitter_id),
    #         tf.strings.join(transmission_id),
    #         tf.strings.join(symbol_index_in_file)
    #     ),
    #     num_parallel_calls=10,
    #     deterministic=True
    # )

    dataset = dataset.map(
        lambda frequency_domain_IQ, day, transmitter_id, transmission_id, symbol_index_in_file: (
            tf.strings.reduce_join(frequency_domain_IQ),
            tf.strings.reduce_join(day),
            tf.strings.reduce_join(transmitter_id),
            tf.strings.reduce_join(transmission_id),
            tf.strings.reduce_join(symbol_index_in_file)
        ),
        num_parallel_calls=10,
        deterministic=True
    )

    # dataset = dataset.map(
    #     lambda x: (
    #          tf.split(x, (symbol_size*batch_size, 1*batch_size, 1*batch_size, 1*batch_size, 8*batch_size))
    #     ),
    #     num_parallel_calls=10,
    #     deterministic=True
    # )

    dataset = dataset.map(
        lambda frequency_domain_IQ,day,transmitter_id,transmission_id,symbol_index_in_file: (
            tf.io.decode_raw(frequency_domain_IQ, tf.float32),
            tf.io.decode_raw(day, tf.uint8),
            tf.io.decode_raw(transmitter_id, tf.uint8),
            tf.io.decode_raw(transmission_id, tf.uint8),
            tf.io.decode_raw(symbol_index_in_file, tf.int64),
        ),
        num_parallel_calls=10,
        deterministic=True
    )

    if batch_size == 1:
        dataset = dataset.map(
            lambda frequency_domain_IQ,day,transmitter_id,transmission_id,symbol_index_in_file: (
                tf.reshape(frequency_domain_IQ, (2,48)),
                tf.reshape(day, () ),
                tf.reshape(transmitter_id, () ),
                tf.reshape(transmission_id, () ),
                tf.reshape(symbol_index_in_file, () ),
            ),
            num_parallel_calls=10,
            deterministic=True
        )
    else:
        dataset = dataset.map(
            lambda frequency_domain_IQ,day,transmitter_id,transmission_id,symbol_index_in_file: (
                tf.reshape(frequency_domain_IQ, (batch_size,2,48)),
                tf.reshape(day, (batch_size) ),
                tf.reshape(transmitter_id, (batch_size) ),
                tf.reshape(transmission_id, (batch_size) ),
                tf.reshape(symbol_index_in_file, (batch_size) ),
            ),
            num_parallel_calls=10,
            deterministic=True
        )

    return dataset

def speed_test(iterable, batch_size=1):
    import time

    last_time = time.time()
    count = 0
    
    for i in iterable:
        count += 1

        if count % int(10000/batch_size) == 0:
            items_per_sec = count / (time.time() - last_time)
            print("Items per second:", items_per_sec*batch_size)
            last_time = time.time()
            count = 0

def check_if_symbol_datasets_are_equivalent(ds1, ds2):
    ds = tf.data.Dataset.zip((ds_orig, ds_new))

    ds = ds.map(
        lambda one, two: (
                tf.math.reduce_all(
                    tf.reshape(
                        tf.math.equal(one[0], two[0]), (96,)
                    )
                ),
                # tf.math.equal(one[0], two[0]), 
                tf.math.equal(one[1], two[1]),
                tf.math.equal(one[2], two[2]),
                tf.math.equal(one[3], two[3]),
                tf.math.equal(one[4], two[4]),
        ),
        num_parallel_calls=10,
        deterministic=True
    )

    ds = ds.map(
        lambda a,b,c,d,e: 
            tf.math.reduce_all(tf.convert_to_tensor((a,b,c,d,e))),
        num_parallel_calls=10,
        deterministic=True
    )

    for e in ds.enumerate():
        if not e[1]:
            print("Datasets not equivalent, differ at index:", e[0])
            return False

    print("Datasets are equivalent")
    return True
    


if __name__ == "__main__":
    ds_orig = vanilla_binary_file_to_symbol_dataset("../bin/day-1_transmitter-11_transmission-1.bin")
    # symbol_dataset_to_file(ds, "t1")
    ds_new  = symbol_dataset_from_file("t1", batch_size=1)

    print(ds_orig.element_spec)
    print(ds_new.element_spec)

    # ds_orig = ds_orig.skip(1)

    # f = None
    # for e in ds_new:
    #     print(e[4])


    check_if_symbol_datasets_are_equivalent(ds_orig, ds_new)