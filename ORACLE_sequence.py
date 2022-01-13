#! /usr/bin/env python3

import pickle
import numpy as np

from steves_utils.ORACLE.utils_v2 import (
    ALL_DISTANCES_FEET,
    ALL_RUNS,
    ALL_SERIAL_NUMBERS,
    filter_paths,
    get_oracle_dataset_path,
    get_oracle_data_files_based_on_criteria,
    serial_number_to_id
)

from steves_utils import file_as_windowed_list


def get_single_oracle_combo(
    desired_serial_number:str,
    desired_run:int,
    desired_distance:int,
    num_floats_in_window:int,
    window_stride:int,
    num_windows:int,
    np_rng=np.random.default_rng(),
    reshape=True
)->np.ndarray:
    paths = get_oracle_data_files_based_on_criteria(
        desired_serial_numbers=[desired_serial_number],
        desired_runs=[desired_run],
        desired_distances=[desired_distance],
    )
    assert len(paths) == 1
    path = paths[0]


    faws = file_as_windowed_list.File_As_Windowed_Sequence(
        path=path,
        window_length=num_floats_in_window,
        stride=window_stride,
        numpy_dtype=np.double,
        return_as_tuple_with_offset=False
    )

    choices = np_rng.choice(len(faws), num_windows, False)
    windows = []
    for c in choices:
        windows.append(faws[c])

    if reshape:
        windows = map(lambda x: x.reshape((2,int(len(x)/2)), order="F"), windows)

    return np.stack(
        list(windows)
    )
    

DESIRED_RUN = 1
NUM_FLOATS_IN_WINDOW=512
WINDOW_STRIDE=50
NUM_WINDOWS=10000
rng = np.random.default_rng(1337)

d = {}
for distance in ALL_DISTANCES_FEET:
    d[distance] = {}
    for serial in ALL_SERIAL_NUMBERS:
        for run in [1]:
            ar = get_single_oracle_combo(
                desired_serial_number=ALL_SERIAL_NUMBERS[0],
                desired_run=DESIRED_RUN,
                desired_distance=ALL_DISTANCES_FEET[0],
                num_floats_in_window=NUM_FLOATS_IN_WINDOW,
                window_stride=WINDOW_STRIDE,
                num_windows=NUM_WINDOWS,
                np_rng=rng
            )

            d[distance][serial] = ar

pickle.dump(d, open("oracle.pkl", "wb"))

    
# import sys
# for distance, serial_val_d in d.items():
#     print(distance)
#     print(serial_val_d.keys())

# sys.exit(1)