#! /usr/bin/env python3

import unittest


from steves_utils.CORES.utils import (
    ALL_DAYS,
    ALL_NODES,
)

from steves_utils.stratified_dataset.stratified_dataset_builder import Stratified_Dataset_Builder_Basic_Test
from create_metehan import Metehan_SDB


Stratified_Dataset_Builder_Basic_Test.SDB = Metehan_SDB()
Stratified_Dataset_Builder_Basic_Test.TEST_DOMAINS = [0,1,2]
Stratified_Dataset_Builder_Basic_Test.TEST_LABELS  = list(range(19))
Stratified_Dataset_Builder_Basic_Test.EXPECTED_X_SHAPE = (2,256)

unittest.main()