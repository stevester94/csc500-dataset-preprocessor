#! /usr/bin/python3
import subprocess
import sys
from typing import List
import json

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

from steves_utils.binary_random_accessor import Binary_OFDM_Symbol_Random_Accessor

pp = pprint.PrettyPrinter(indent=4)
pprint = pp.pprint

def pool_worker_init(paths):
    global BOSRA
    BOSRA = Binary_OFDM_Symbol_Random_Accessor(paths)

def pool_worker_process(index):
    global BOSRA
    return BOSRA[index]

class BinarySymbolRandomAccessor():
    def __init__(
        self,
        seed,
        batch_size,
        bin_path="/mnt/wd500GB/CSC500/csc500-super-repo/csc500-dataset-preprocessor/bin",
        day_to_get="All",
        transmitter_id_to_get="All",
        transmission_id_to_get="All",
        num_workers=10,
        randomize=True
    ):
        self.day_to_get = day_to_get
        self.transmitter_id_to_get = transmitter_id_to_get
        self.transmission_id_to_get = transmission_id_to_get
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.randomize = randomize

        self.rng = np.random.default_rng(self.seed)

        self.paths = self.filter_datasets(self.get_binaries_in_dir(bin_path))

        self.paths.sort()

        if len(self.paths) == 0:
            print("No paths remained after filtering. Time to freak out!")
            sys.exit(1)

        if randomize:
            self.rng.shuffle(self.paths)
        
        self.bosra = Binary_OFDM_Symbol_Random_Accessor(self.paths, max_file_descriptors=0)
        self.cardinality = self.bosra.get_cardinality()

        # We generate our own indices based on the seed
        print("Generating our indices")
        indices = np.arange(0, self.cardinality)

        if randomize:
            print("Randomizing our indices")
            self.rng.shuffle(indices)
        self.indices = indices

        print("Ready")

    def get_binaries_in_dir(self, path):
        (_, _, filenames) = next(os.walk(path))
        return [os.path.join(path,f) for f in filenames if ".bin" in f]

    def is_any_word_in_string(self, list_of_words, string):
        for w in list_of_words:
            if w in string:
                return True
        return False

    def filter_datasets(self, paths):
        filtered_paths = paths
        if self.day_to_get != "All":
            assert(isinstance(self.day_to_get, list))
            assert(len(self.day_to_get) > 0)
            
            filt = ["day-{}_".format(f) for f in self.day_to_get]
            filtered_paths = [p for p in filtered_paths if self.is_any_word_in_string(filt, p)]

        if self.transmitter_id_to_get != "All":
            assert(isinstance(self.transmitter_id_to_get, list))
            assert(len(self.transmitter_id_to_get) > 0)

            filt = ["transmitter-{}_".format(f) for f in self.transmitter_id_to_get]
            filtered_paths = [p for p in filtered_paths if self.is_any_word_in_string(filt, p)]

        if self.transmission_id_to_get != "All":
            assert(isinstance(self.transmission_id_to_get, list))
            assert(len(self.transmission_id_to_get) > 0)

            filt = ["transmission-{}.".format(f) for f in self.transmission_id_to_get]
            filtered_paths = [p for p in filtered_paths if self.is_any_word_in_string(filt, p)]

        return filtered_paths


    def get_paths(self):
        return self.paths
    
############################################

    def get_total_dataset_cardinality(self):
        return self.cardinality

    def _index_generator(self, indices, repeat, shuffle):
        while True:
            if shuffle:
                self.rng.shuffle(indices)

            yield from indices

            if not repeat:
                return

    def all_generator(self, repeat=False, shuffle=True):
        with mp.Pool(self.num_workers, pool_worker_init, (self.paths,)) as worker_pool:
            pool_imap = worker_pool.imap(pool_worker_process, self._index_generator(self.indices, repeat, self.randomize))
            yield from self.batch_generator_from_generator(pool_imap, self.batch_size)

    # Drops remainder that will not fit in batch_size
    def batch_generator_from_generator(self, gen, batch_size):
        exhausted = False
        while True:
            frequency_domain_IQ = []
            day = []
            transmitter_id = []
            transmission_id = []
            symbol_index_in_file = []
            for i in range(batch_size):
                try:
                    e = next(gen)
                except StopIteration:
                    return

                frequency_domain_IQ.append(  e["frequency_domain_IQ"])
                day.append(                  e["day"])
                transmitter_id.append(       e["transmitter_id"])
                transmission_id.append(      e["transmission_id"])
                symbol_index_in_file.append( e["symbol_index"])

       
            yield (
                tf.convert_to_tensor(frequency_domain_IQ, dtype=tf.float32),
                tf.convert_to_tensor(day, dtype=tf.uint8),
                tf.convert_to_tensor(transmitter_id, dtype=tf.uint8),
                tf.convert_to_tensor(transmission_id, dtype=tf.uint8),
                tf.convert_to_tensor(symbol_index_in_file, dtype=tf.int64),
            )
    

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





def convert(
    seed,
    batch_size,
    day_to_get,
    transmitter_id_to_get,
    transmission_id_to_get,
    randomize,
    out_dir,
    max_file_size_Bytes):
    bsda = BinarySymbolRandomAccessor(
        seed=seed,
        batch_size=batch_size,
        day_to_get=day_to_get,
        transmitter_id_to_get=transmitter_id_to_get,
        transmission_id_to_get=transmission_id_to_get,
        randomize=randomize
    )

    gen = bsda.all_generator()

    current_file_size = 0
    current_file_index = 0
    f = open("{}/dataset_batch-{}_part-{}.bin".format(out_dir, batch_size, current_file_index), "wb")

    for e in gen:
        frequency_domain_IQ, day, transmitter_id, transmission_id, symbol_index_in_file = e

        members = [frequency_domain_IQ, day, transmitter_id, transmission_id, symbol_index_in_file]

        members_as_numpy = [m.numpy() for m in members]
        members_as_buffer = [m_np.tobytes() for m_np in members_as_numpy]

        total_length = sum([len(x) for x in members_as_buffer])

        assert(total_length <= max_file_size_Bytes)

        if current_file_size + total_length > max_file_size_Bytes:
            f.close()
            current_file_index += 1
            current_file_size = 0
            f = open(out_dir + "/dataset_batch-{}_part-{}.bin".format(batch_size, current_file_index), "wb")

        
        for buffer in members_as_buffer:
            f.write(buffer)

        current_file_size += total_length

original_ds = None
fldsa_ds = None
original_offender = None
new_offender  = None
def test():
    global original_ds
    global fldsa_ds
    global original_offender
    global new_offender

    from steves_utils.fixed_length_accessor import FixedLengthDatasetAccessor
    from steves_utils.datasetaccessor import SymbolDatasetAccessor

    # convert(
    #     seed=1337,
    #     batch_size=1000,
    #     day_to_get=[1],
    #     transmitter_id_to_get=[10,11],
    #     transmission_id_to_get=[1],
    #     randomize=False,
    #     out_dir="./test/",
    #     max_file_size_Bytes=10e6
    # )

    # Use this just for the paths
    bsda = BinarySymbolRandomAccessor(
        seed=1337,
        batch_size=1000,
        day_to_get=[1],
        transmitter_id_to_get=[10,11],
        transmission_id_to_get=[1],
        randomize=False
    )

    flda = FixedLengthDatasetAccessor(bin_path="./test/")

    original_ds   =  tf.data.FixedLengthRecordDataset(
        bsda.paths, record_bytes=384, header_bytes=None, footer_bytes=None, buffer_size=None,
        compression_type=None, num_parallel_reads=1
    )

    fldsa_ds = flda.get_dataset().unbatch()

    zipped = tf.data.Dataset.zip((original_ds, fldsa_ds))

    index = 0
    for e in zipped:

        original_as_numpy = e[0].numpy()
        new_as_numpy      = e[1][0].numpy().tobytes(order="F")

        if original_as_numpy != new_as_numpy:
            print(original_as_numpy)
            print(new_as_numpy)
            original_offender = original_as_numpy
            new_offender = e
            raise Exception("Not Equivalent, offender at {}".format(index))

        index += 1


            
    


if __name__ == "__main__":
    if sys.argv[1] == "convert":
        convert(
            seed=1337,
            batch_size=1000,
            day_to_get=[1],
            transmitter_id_to_get=[10,11],
            transmission_id_to_get=[1],
            randomize=False,
            out_dir="./test/",
            max_file_size_Bytes=10e6
        )
    elif sys.argv[1] == "test": test()
    else: print("usage: convert | test")
        

    # speed_test(gen, 1000)


