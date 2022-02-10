#! /usr/bin/env python3

import pickle
import os
import numpy as np
from steves_utils.utils_v2 import get_datasets_base_path
from steves_utils.stratified_dataset.stratified_dataset_builder import Stratified_Dataset_Builder


class Metehan_SDB(Stratified_Dataset_Builder):
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
    def __init__(
        self,
        simulations_npz_path:str=os.path.join(get_datasets_base_path(), "simulations.npz"),
        num_complex_samples_per_example=256
    ) -> None:
        super().__init__()
        self.simulations_npz_path = simulations_npz_path
        self.num_complex_samples_per_example = num_complex_samples_per_example

    def build_dataset(self, seed: int, domains: list, labels: list, out_path: str):
        rng = np.random.default_rng(seed)
        d = {}

        """
        This one is a little funky because there really are not domains
        That being said, we'll split their train, val, test into domains {0,1,2}
        """

        npz = np.load(self.simulations_npz_path)

        # This is such a small file I'm going to be incredibly lazy

        if 0 in domains:
            d[0] = {}
            for i in range(19): d[0][i] = []
            for x,y in zip(npz["arr_0"], np.argmax(npz["arr_1"], axis=1)):
                if y in labels:
                    d[0][y].append(x[:self.num_complex_samples_per_example].T)

        if 1 in domains:
            d[1] = {}
            for i in range(19): d[1][i] = []
            for x,y in zip(npz["arr_2"], np.argmax(npz["arr_3"], axis=1)):
                if y in labels:
                    d[1][y].append(x[:self.num_complex_samples_per_example].T)

        if 2 in domains:
            d[2] = {}
            for i in range(19): d[2][i] = []
            for x,y in zip(npz["arr_4"], np.argmax(npz["arr_5"], axis=1)):
                if y in labels:
                    d[2][y].append(x[:self.num_complex_samples_per_example].T)
            
            
        
        for u, y_X_dict in d.items():
            for y, X in y_X_dict.items():
                rng.shuffle(X)
                y_X_dict[y] = np.stack(X)

        metadata = {
            "seed": seed,
            "comment": "",
            "addenda": None
        }

        out = {
            "data": d,
            "metadata": metadata
        }

        with open(out_path, "wb") as f:
            pickle.dump(out, f)
            
if __name__ == "__main__":
    metehan_sdb = Metehan_SDB()
    metehan_sdb.build_dataset(
        seed=1337,
        domains=[0,1,2],
        labels=list(range(20)),
        out_path="metehan.stratified_ds.2022A.pkl"
    )