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

from steves_utils.utils_v2 import to_hash
from create_oracle import generate_pickle

class Test_ORACLE_Pickle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        PICKLE_PATH="/tmp/oracle_test.pkl"

        cls.desired_labels = ALL_SERIAL_NUMBERS
        cls.desired_domains = ALL_DISTANCES_FEET
        cls.desired_seed    = 1337

        cls.num_floats_in_window = 512
        cls.window_stride        = 50
        cls.num_windows          = 100

        generate_pickle(
            serial_numbers=cls.desired_labels,
            runs=[1],
            distances=cls.desired_domains,
            num_floats_in_window=cls.num_floats_in_window,
            window_stride=cls.window_stride,
            num_windows=cls.num_windows,
            seed=cls.desired_seed,
            out_path=PICKLE_PATH,
        )

        with open(PICKLE_PATH, "rb") as f:
            cls.d = pickle.load(f)

    def test_unique_hashes(self):
        data = self.d["data"]
        metadata = self.d["metadata"]
        h = []

        for domain, label_and_x_d in data.items():
            for label, all_x in label_and_x_d.items():
                for window in all_x:
                    h.append(
                        to_hash(window)
                    )
        
        self.assertEqual(
            len(h),
            len(set(h))
        )

    def test_desired_domains(self):
        data = self.d["data"]
        metadata = self.d["metadata"]
        domains = metadata["distances"]

        self.assertEqual(
            set(domains),
            set(self.desired_domains)
        )

    def test_desired_labels(self):
        data = self.d["data"]
        metadata = self.d["metadata"]

        labels = set()

        for domain, label_and_x_d in data.items():
            for label, all_x in label_and_x_d.items():
                labels.add(label)

        self.assertEqual(
            labels,
            set(self.desired_labels)
        )

    def test_num_examples_and_shape(self):
        data = self.d["data"]
        metadata = self.d["metadata"]

        for domain, label_and_x_d in data.items():
            for label, all_x in label_and_x_d.items():
                self.assertEqual(
                    all_x.shape,
                    (self.num_windows, 2, int(self.num_floats_in_window/2))
                )


unittest.main()