#! /usr/bin/python3
import sys
import re
import numpy as np

import datasetaccessor
import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras as keras

original_record_base_path="../bin/"
symbol_tfrecords_base_path="../symbol_tfrecords"

if __name__ == "__main__":
    if len(sys.argv) not in [2, 4]:
        print("Usage: <tfdataset path>|<day> <transmitter_id> <transmission id>")
        sys.exit(1)


    if len(sys.argv) == 2:
        match  = re.search("day-([0-9]+)_transmitter-([0-9]+)_transmission-([0-9]+)", sys.argv[1])
        (day, transmitter_id, transmission_id) = match.groups()
        if match == None:
            print("Malformed argument, exiting")
            sys.exit(1)

    if len(sys.argv) == 4:
        day = sys.argv[1]
        transmitter_id =sys.argv[2]
        transmission_id = sys.argv[3]

    original_record_path = original_record_base_path + "/day-{day}_transmitter-{transmitter_id}_transmission-{transmission_id}.bin".format(
            day=day, transmitter_id=transmitter_id, transmission_id=transmission_id)

    print("Processing:", original_record_path)

    vdsa = datasetaccessor.SymbolDatasetAccessor(
        day_to_get=[day],
        transmitter_id_to_get=[transmitter_id],
        transmission_id_to_get=[transmission_id],
        tfrecords_path=symbol_tfrecords_base_path)

    ds = vdsa.get_dataset().prefetch(10000)


    with open(original_record_path, "rb") as f:
        buf = f.read()
        original_array = np.frombuffer(buf, dtype=np.single)

    ds = ds.map(lambda e: ( tf.reshape(
            tf.stack( [e["frequency_domain_IQ"][0], e["frequency_domain_IQ"][1]] , axis=1),
            [-1, 96]),
        e["symbol_index"] ),
        num_parallel_calls=10,
        deterministic=False).prefetch(10000)

    elements_checked = 0
    for e in ds:
        elements_checked += 96
        assert(
            np.array_equal( original_array[e[1]*96:e[1]*96+96], e[0][0] )
        )

    assert(elements_checked == len(original_array) )

    

    # even_new = tf.constant([0,2,4])
    # odd_new = tf.constant([1,3,5])
    # print(even_new.shape)

    # t = tf.stack([even_new, odd_new], axis=1)
    # print(t)

            # np.array_equal( original_array[e["symbol_index"]*96:e["symbol_index"]*96+96:2], e["frequency_domain_IQ"][0]) and
            # np.array_equal( original_array[e["symbol_index"]*96+1:e["symbol_index"]*96+96:2], e["frequency_domain_IQ"][1])

    # print(t)

    print("All checks passed")
