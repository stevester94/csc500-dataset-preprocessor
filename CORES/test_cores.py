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

generate_pickle(
    seed=1337,
    days=ALL_DAYS,
    nodes=ALL_NODES,
    out_path="/tmp/cores.pkl"
)

d = pickle.load(open("/tmp/cores.pkl", "rb"))

def numpy_to_hash(n:np.ndarray):
    return hash(n.data.tobytes())

class Test_CORES_Pickle(unittest.TestCase):
    def test_unique_hashes(self):
        h = []

        for day, serial_iq_d in d.items():
            for node, all_iq in serial_iq_d.items():
                for window in all_iq:
                    h.append(
                        numpy_to_hash(window)
                    )
        
        self.assertEqual(
            len(h),
            len(set(h))
        )

unittest.main()