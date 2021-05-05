#! /usr/bin/python3

from steves_utils import utils

import sys


if __name__ == "__main__":

    in_dir = sys.argv[1]

    paths = utils.get_files_with_suffix_in_dir(in_dir, ".bin")




    top = [[] for _ in range(10)]
    for day in top:
        day.extend([[] for _ in range(21)])

    for p in paths:
        m = utils.metadata_from_path(p)


        top[m["day"]][m["transmitter_id"]].append(m["transmission_id"])


    for did, day in enumerate(top):
        for tid, transmitter in enumerate(day):
            if len(transmitter) != 0:
                print("Day", did, "Transmitter", tid)
                print("Transmissions", transmitter)

    for did, day in enumerate(top):
        for tid, transmitter in enumerate(day):
            if len(transmitter) != 0:
                print("day-{}_transmitter-{}_transmission-{}.bin".format(did, tid, transmitter[0]))

    # print(top)