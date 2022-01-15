#! /usr/bin/env python3

from importlib.metadata import metadata
import pickle
import os
import numpy as np

DATASET_PATH = "/mnt/wd500GB/CSC500/csc500-super-repo/datasets/CORES/orbit_rf_identification_dataset_updated"


from steves_utils.CORES.utils import (
    ALL_DAYS,
    dataset_day_name_mapping,
    ALL_NODES ,
)


# CORES is small enough that we just get it all
def generate_pickle(
    seed:int,
    days:list,
    nodes:list,
    out_path:str,
):
    rng = np.random.default_rng(seed)
    d = {}
    for day in days:
        d[day] = {}
        dataset_path = os.path.join(DATASET_PATH, dataset_day_name_mapping[day])
        with open(dataset_path,'rb') as f:
            ds = pickle.load(f)

        for node in nodes:
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
        "days": days,
        "nodes": nodes,
    }

    out = {
        "data": d,
        "metadata": metadata
    }

    with open(out_path, "wb") as f:
        pickle.dump(out, f)

            
if __name__ == "__main__":
    generate_pickle(
        seed=1337,
        days=ALL_DAYS,
        nodes=ALL_NODES,
        out_path="cores.pkl"
    )