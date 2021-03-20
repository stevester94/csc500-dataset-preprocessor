#! /usr/bin/python3

import sys

import pprint
pp = pprint.PrettyPrinter(indent=4)
_print = print
print = pp.pprint

def gen_sha_name_list(path):
    l = []

    with open(path) as f:
        for line in f:
            sha_name_tup = line.rstrip().split()
            
            l.append(sha_name_tup)

            assert(len(sha_name_tup) == 2)

    return l

def check_for_duplicates(l):
    seen_shas = {}

    for sha, name in l:
        if sha not in seen_shas:
            seen_shas[sha] = [name]
        else:
            seen_shas[sha] += [name]


    duples = {}

    for sha, names in seen_shas.items():
        if len(names) > 1:
            duples[sha] = names

    return duples


bin_sha_path = sys.argv[1]
metadata_sha_path = sys.argv[2]

bin_list        = gen_sha_name_list(bin_sha_path)
metadata_list   = gen_sha_name_list(metadata_sha_path)

#for l in metadata_list: _print(l[1])
bin_dupes = check_for_duplicates(bin_list)

for sha, names in bin_dupes.items():
    for n in names:
        _print(n, end=' ')
    _print()

bin_shas_only = [b[0] for b in bin_list]

for m in metadata_list:
    if m[0] not in bin_shas_only:
        _print(m[0], m[1])


metadata_sha_only = [m[0] for m in metadata_list]
for b in bin_list:
    if b[0] not in metadata_sha_only:
        _print(b[0], b[1])


bin_shas_only.sort()
metadata_sha_only.sort()
if bin_shas_only == metadata_sha_only:
    print("We're good")
else:
    print("We're NOT good")