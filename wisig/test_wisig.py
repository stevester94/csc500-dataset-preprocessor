#! /usr/bin/env python3

import unittest


from steves_utils.wisig.utils import (
    ALL_DAYS,
    day_name_mapping,
    ALL_NODES_MINIMUM_100_EXAMPLES
)

from steves_utils.stratified_dataset.stratified_dataset_builder import Stratified_Dataset_Builder_Basic_Test
from create_wisig import Wisig_SDB


Stratified_Dataset_Builder_Basic_Test.SDB = Wisig_SDB()
Stratified_Dataset_Builder_Basic_Test.TEST_DOMAINS = ALL_DAYS
Stratified_Dataset_Builder_Basic_Test.TEST_LABELS  = ALL_NODES_MINIMUM_100_EXAMPLES
Stratified_Dataset_Builder_Basic_Test.EXPECTED_X_SHAPE = (2,256)

unittest.main()