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

def generate_pickle(
    serial_numbers,
    runs,
    distances,
    num_floats_in_window,
    window_stride,
    num_windows,
    seed,
    out_path,
):
    d = {}
    for distance in distances:
        d[distance] = {}
        for serial in serial_numbers:
            for run in runs:
                ar = get_single_oracle_combo(
                    desired_serial_number=serial,
                    desired_run=run,
                    desired_distance=distance,
                    num_floats_in_window=num_floats_in_window,
                    window_stride=window_stride,
                    num_windows=num_windows,
                    np_rng=np.random.default_rng(seed)
                )

                d[distance][serial] = ar

    pickle.dump(d, open(out_path, "wb"))

if __name__ == "__main__":
    generate_pickle(
        serial_numbers=ALL_SERIAL_NUMBERS,
        runs=[1],
        distances=ALL_DISTANCES_FEET,
        num_floats_in_window=512,
        window_stride=50,
        num_windows=10000,
        seed=1337,
        out_path="oracle.pkl",
    )