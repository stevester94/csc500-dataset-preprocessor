#! /usr/bin/env python3

import pickle
import numpy as np
import unittest

from create_cores import generate_pickle

from steves_utils.CORES.utils import (
    ALL_DAYS,
    dataset_day_name_mapping,
    ALL_NODES ,
)

from steves_utils.utils_v2 import to_hash




class Test_CORES_Pickle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.desired_domains = ALL_DAYS
        cls.desired_labels  = ALL_NODES
        cls.desired_seed    = 1337

        generate_pickle(
            seed=cls.desired_seed,
            days=cls.desired_domains,
            nodes=cls.desired_labels,
            out_path="/tmp/cores.pkl"
        )

        with open("/tmp/cores.pkl", "rb") as f:
            cls.d = pickle.load(f)

    def test_unique_hashes(self):
        data = self.d["data"]
        metadata = self.d["metadata"]
        h = []

        for day, serial_iq_d in data.items():
            for node, all_iq in serial_iq_d.items():
                for window in all_iq:
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
        domains = metadata["days"]

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
                    all_x.shape[1:],
                    (2, 256)
                )

unittest.main()