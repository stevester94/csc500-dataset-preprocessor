#! /usr/bin/env python3

import pickle
import os
import numpy as np
from steves_utils.wisig.utils import get_wisig_dataset_path
from steves_utils.stratified_dataset.stratified_dataset_builder import Stratified_Dataset_Builder

from steves_utils.wisig.utils import (
    ALL_DAYS,
    day_name_mapping,
    ALL_NODES_MINIMUM_100_EXAMPLES
)

class Wisig_SDB(Stratified_Dataset_Builder):
    def __init__(
        self,
        dataset_base_path:str=get_wisig_dataset_path()
    ) -> None:
        super().__init__()
        self.dataset_base_path = dataset_base_path

    def build_dataset(self, seed: int, domains: list, labels: list, out_path: str):
        rng = np.random.default_rng(seed)
        d = {}
        for day in domains:
            d[day] = {}
            dataset_path = os.path.join(self.dataset_base_path, day_name_mapping[day])
            with open(dataset_path,'rb') as f:
                ds = pickle.load(f)

            for node in labels:
                d[day][node] = []

                # data and node_list are parallel
                i = ds["node_list"].index(node)
                data = ds["data"][i]
                for x in rng.choice(data, len(data), replace=False):
                        """
                        It's just a transpose since the data is originally saved as "rows"
                        IE shape is originally 256,2
                        But we want 2,256
                        """
                        d[day][node].append(x.T)
                
                # Stack it all into one big np array
                d[day][node] = np.stack(d[day][node])
        

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
    wisig_sdb = Wisig_SDB()
    wisig_sdb.build_dataset(
        seed=1337,
        domains=ALL_DAYS,
        labels=ALL_NODES_MINIMUM_100_EXAMPLES,
        out_path="wisig.node3-19.stratified_ds.2022A.pkl"
    )