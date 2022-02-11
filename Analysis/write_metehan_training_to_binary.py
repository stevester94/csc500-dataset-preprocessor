#! /usr/bin/env python3

import os
import numpy as np
from steves_utils.utils_v2 import get_datasets_base_path

"""
simulations.npz is arranged as such
Train: arr_0 (3800, 3200, 2)
Train: arr_1 (3800, 19)
Val: arr_2 (1900, 3200, 2)
Val: arr_3 (1900, 19)
Test: arr_4 (1900, 3200, 2)
Test: arr_5 (1900, 19)

(I have no idea what these are supposed to be)
arr_6 (3800,)
arr_7 (1900,)
arr_8 (1900,)

"""

simulations_npz_path=os.path.join(get_datasets_base_path(), "simulations.npz")


npz = np.load(simulations_npz_path)

t = npz["arr_0"][0].T
c = np.empty((t[0].size + t[1].size,), dtype=t.dtype)
c[0::2] = t[0]
c[1::2] = t[1]

assert(t[0][0] == c[0])
assert(t[1][0] == c[1])

print(t.dtype)

c.astype('float64').tofile("metehan.bin")