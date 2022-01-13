#! /usr/bin/env python3

import pickle
import numpy as np
import unittest

from steves_utils.ORACLE.utils_v2 import (
    ALL_DISTANCES_FEET,
    ALL_RUNS,
    ALL_SERIAL_NUMBERS,
    NUMBER_OF_ORIGINAL_FILE_PAIRS,
    filter_paths,
    get_oracle_dataset_path,
    get_oracle_data_files_based_on_criteria,
    serial_number_to_id
)


from create_oracle import generate_pickle

generate_pickle(
    serial_numbers=ALL_SERIAL_NUMBERS,
    runs=[1],
    distances=ALL_DISTANCES_FEET,
    num_floats_in_window=512,
    window_stride=50,
    num_windows=100,
    seed=1337,
    out_path="/tmp/oracle_test.pkl",
)

d = pickle.load(open("/tmp/oracle_test.pkl", "rb"))

def numpy_to_hash(n:np.ndarray):
    return hash(n.data.tobytes())

class Test_ORACLE_Pickle(unittest.TestCase):
    def test_unique_hashes(self):
        h = []

        for distance, serial_iq_d in d.items():
            for serial, all_iq in serial_iq_d.items():
                for window in all_iq:
                    h.append(
                        numpy_to_hash(window)
                    )
        
        self.assertEqual(
            len(h),
            len(set(h))
        )

unittest.main()