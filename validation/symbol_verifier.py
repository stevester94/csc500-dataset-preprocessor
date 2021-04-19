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

    ds = vdsa.get_dataset()

    last_index = -1
    reconstructed_tensor = []

    original_index = 0

    for e in ds:
        index = e["symbol_index"].numpy()
        assert(index == last_index+1)
        last_index = index
        assert(e["frequency_domain_IQ"].shape == (2,48))

        t = e["frequency_domain_IQ"]
        c = np.empty((t[0].shape[0] + t[1].shape[0]), dtype=np.single)
        c[0::2] = t[0]
        c[1::2] = t[1]
        
        reconstructed_tensor.extend(c)

    
    with open(original_record_path, "rb") as f:
        buf = f.read()
        original_array = np.frombuffer(buf, dtype=np.single)

    reconstructed_array = np.array(reconstructed_tensor, dtype=np.single)

    print(reconstructed_array)
    print(original_array)

    assert(np.array_equal(original_array, reconstructed_array))

    print("All checks passed")
