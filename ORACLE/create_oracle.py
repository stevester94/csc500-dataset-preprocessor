#! /usr/bin/env python3

import pickle
from random import seed
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

from steves_utils.stratified_dataset.stratified_dataset_builder import Stratified_Dataset_Builder

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

class ORACLE_SDB(Stratified_Dataset_Builder):
    def __init__(
        self,
        num_floats_in_window:int,
        window_stride:int,
        num_windows:int,
        runs:list,
    ) -> None:
        super().__init__()
        self.num_floats_in_window = num_floats_in_window
        self.window_stride        = window_stride
        self.num_windows          = num_windows
        self.runs                 = runs
    
    def build_dataset(self, seed: int, domains: list, labels: list, out_path: str):
        d = {}
        rng = np.random.default_rng(seed)
        for distance in domains:
            d[distance] = {}
            for serial in labels:
                for run in self.runs:
                    ar = get_single_oracle_combo(
                        desired_serial_number=serial,
                        desired_run=run,
                        desired_distance=distance,
                        num_floats_in_window=self.num_floats_in_window,
                        window_stride=self.window_stride,
                        num_windows=self.num_windows,
                        np_rng=rng
                    )

                    d[distance][serial] = ar

        metadata = {
            "runs": self.runs,
            "window_stride": self.window_stride,
            "num_windows": self.num_windows,
            "seed": seed,
        }
        out = {
            "metadata": metadata,
            "data": d
        }

        with open(out_path, "wb") as f:
            pickle.dump(out, f)



if __name__ == "__main__":
    oracle_sdb = ORACLE_SDB(
        num_floats_in_window=512,
        window_stride=50,
        num_windows=10000,
        runs=[1],
    )
    oracle_sdb.build_dataset(
        seed=1337,
        domains=list(set(ALL_DISTANCES_FEET)-{2,62,56}),
        labels=ALL_SERIAL_NUMBERS,
        out_path="oracle.Run1_10kExamples_stratified_ds.2022A.pkl",
    )


    oracle_sdb = ORACLE_SDB(
        num_floats_in_window=512,
        window_stride=50,
        num_windows=10000,
        runs=[2],
    )
    oracle_sdb.build_dataset(
        seed=1337,
        domains=list(set(ALL_DISTANCES_FEET)-{2,62,56}),
        labels=ALL_SERIAL_NUMBERS,
        out_path="oracle.Run2_10kExamples_stratified_ds.2022A.pkl",
    )