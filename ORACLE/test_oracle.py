#! /usr/bin/env python3

import unittest

from steves_utils.ORACLE.utils_v2 import (
    ALL_DISTANCES_FEET,
    ALL_SERIAL_NUMBERS,
)

from steves_utils.stratified_dataset.stratified_dataset_builder import Stratified_Dataset_Builder_Basic_Test
from create_oracle import ORACLE_SDB

oracle_sdb = ORACLE_SDB(
    num_floats_in_window=512,
    window_stride=50,
    num_windows=100,
    runs=[1],
)

Stratified_Dataset_Builder_Basic_Test.SDB = oracle_sdb
Stratified_Dataset_Builder_Basic_Test.TEST_DOMAINS = ALL_DISTANCES_FEET
Stratified_Dataset_Builder_Basic_Test.TEST_LABELS  = ALL_SERIAL_NUMBERS
Stratified_Dataset_Builder_Basic_Test.EXPECTED_X_SHAPE = (2,256)

unittest.main()