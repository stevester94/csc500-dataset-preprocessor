#! /usr/bin/python3

import os
import sys
import re

import pprint
pp = pprint.PrettyPrinter(indent=4)
pprint = pp.pprint



# Will only get files in the path, will not recurse to sub-dirs
def get_tfrecords_in_dir(path):
    (_, _, filenames) = next(os.walk(path))
    return [os.path.join(path,f) for f in filenames if ".tfrecord" in f]

files = []
if __name__ == "__main__":
    records = get_tfrecords_in_dir("../vanilla_tfrecords")

    for r in records:
        match = re.search("day-([0-9]+)_transmitter-([0-9]+)_transmission-([0-9]+)", r)

        if match == None:
            print("No matches found!")
            print(r)
            sys.exit(1)

        (day, transmitter, transmission) = match.groups()
        files.append({
            "day": int(day),
            "transmission": int(transmission),
            "transmitter": int(transmitter)
        })

    pprint(files)

    all_days = (set([f["day"] for f in files]))
    print("All Days:", all_days)

    all_transmitters = (set([f["transmitter"] for f in files]))
    print("All Transmitters: ", all_transmitters)

    all_transmissions = (set([f["transmission"] for f in files]))
    print("All Transmissions: ", all_transmissions)


    for transmitter in all_transmitters:
        days_for_this_transmitter = set([int(f["day"]) for f in files if f["transmitter"] == transmitter])

        print(days_for_this_transmitter)
        if days_for_this_transmitter == all_days:
            print("Transmitter", transmitter, "is in all days")


    for transmitter in all_transmitters:
        transmissions_for_this_transmitter_in_days_subset = set([int(f["transmission"]) for f in files if f["transmitter"] == transmitter and f["day"] in [2,3,4]])

        print("Transmitter:", transmitter, "Transmissions:", transmissions_for_this_transmitter_in_days_subset )
