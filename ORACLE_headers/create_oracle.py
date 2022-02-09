#! /usr/bin/env python3

from email import header
import pickle
from tkinter.tix import MAX
import numpy as np
import json

from numpy import indices

from steves_utils.ORACLE.utils_v2 import (
    ALL_DISTANCES_FEET,
    ALL_RUNS,
    ALL_SERIAL_NUMBERS,
    filter_paths,
    get_oracle_dataset_path,
    get_oracle_data_files_based_on_criteria,
    serial_number_to_id
)


"""
Indices are the beginning of each 802.11a in the ORACLE dataset
These indices are the index of the COMPLEX SAMPLE, NOT THE DOUBLE SAMPLE
Meaning since we are indexing into a numpy array of double, we must multiple the index
in indices by 2
"""
def get_single_oracle_combo(
    desired_serial_number:str,
    desired_run:int,
    desired_distance:int,
    header_indices:dict,
    num_floats_in_window:int,
    num_windows:int,
    np_rng=np.random.default_rng(),
    reshape=True,
)->np.ndarray:
    paths = get_oracle_data_files_based_on_criteria(
        desired_serial_numbers=[desired_serial_number],
        desired_runs=[desired_run],
        desired_distances=[desired_distance],
    )
    assert len(paths) == 1
    path = paths[0]

    assert path in header_indices
    indices = header_indices[path]

    mm = np.memmap(path, np.double)

    print(f"Processing {path}: choosing {num_windows} from a possible {len(indices)} indices")

    choices = np_rng.choice(len(indices), num_windows, False)
    windows = []
    for c in choices:
        start = c*2
        ar = np.array(mm[start:start+num_floats_in_window]) 
        assert len(ar) == num_floats_in_window
        windows.append( ar )

    if reshape:
        windows = map(lambda x: x.reshape((2,int(len(x)/2)), order="F"), windows)

    return np.stack(
        list(windows)
    )

def sanity_check_header_indices(header_indices:dict, paths:list)->None:
    MAX_DIFFERENCE = 100000
    MIN_INDICES = 1000

    for path in paths:
        indices = header_indices[path]
        if len(indices) < MIN_INDICES:
            raise Exception(f"Path {path} only has {len(indices)} indices (Minimum is {MIN_INDICES})")

        i = 0
        while True:
            if not isinstance(indices, list):
                raise Exception(f"Path {path} is not a list (got {type(indices)})")
                
                
            if (i == len(indices)) or (i + 1 == len(indices)):
                break

            """
            Check if the headers are too far apart
            """
            diff = indices[i+1] - indices[i]
            if diff > MAX_DIFFERENCE:
                raise Exception(f"Path {path} has indices that are separated farther than the allowed of {MAX_DIFFERENCE} (got {diff})")
            
            


            i += 1

def generate_pickle(
    serial_numbers,
    runs,
    distances,
    num_floats_in_window,
    header_indices_path,
    num_windows,
    seed,
    out_path,
):
    d = {}
    rng = np.random.default_rng(seed)

    with open(header_indices_path, "r") as f:
        header_indices = json.load(f)

    # Begin sanity checking of the indices
    paths = get_oracle_data_files_based_on_criteria(
        desired_serial_numbers=serial_numbers,
        desired_runs=runs,
        desired_distances=distances,
    )
    assert len(paths) > 0
    sanity_check_header_indices(header_indices, paths)

    for distance in distances:
        d[distance] = {}
        for serial in serial_numbers:
            for run in runs:
                ar = get_single_oracle_combo(
                    desired_serial_number=serial,
                    desired_run=run,
                    desired_distance=distance,
                    num_floats_in_window=num_floats_in_window,
                    header_indices=header_indices,
                    num_windows=num_windows,
                    np_rng=rng
                )

                d[distance][serial] = ar

    metadata = {
        "serial_numbers": serial_numbers,
        "runs": runs,
        "distances": distances,
        "num_floats_in_window": num_floats_in_window,
        "num_windows": num_windows,
        "seed": seed,
    }
    out = {
        "metadata": metadata,
        "data": d
    }

    with open(out_path, "wb") as f:
        pickle.dump(out, f)

if __name__ == "__main__":
    generate_pickle(
        serial_numbers=ALL_SERIAL_NUMBERS,
        runs=[1],
        distances=list(set(ALL_DISTANCES_FEET)-{2,62}),
        num_floats_in_window=512,
        header_indices_path="./isolate_headers/indices.json",
        num_windows=1365,
        seed=1337,
        out_path="oracle.stratified_ds.2022A.pkl",
    )