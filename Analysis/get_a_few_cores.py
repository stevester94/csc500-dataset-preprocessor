#! /usr/bin/env python3

from steves_utils.simple_datasets.CORES.dataset_accessor import get_datasets
from struct import *
from steves_utils.CORES.utils import (
    ALL_NODES,
    ALL_DAYS
)

train, val, test = get_datasets(
    nodes=ALL_NODES,
    days=ALL_DAYS,
    num_examples_per_day_per_node=100,
)


x,y,u = next(iter(train))

print(x.shape)

X = []



SKIP_COUNT = 11
train_iter = iter(train)

for _ in range(SKIP_COUNT):
    next(train_iter)

X, _, _ = next(train_iter)

flatten = []
for i,q in zip(x[0], x[1]):
    flatten.append(i)
    flatten.append(q)

X = flatten

b = pack("512d", *X)

# print(len(X))

with open("cores_one.bin", "wb") as f:
    f.write(b)