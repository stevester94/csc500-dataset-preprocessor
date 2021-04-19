#! /usr/bin/bash

tmp1=$(mktemp)
tmp2=$(mktemp)
find ../bin | grep -E '\.bin$' | sort > $tmp1
find ../tfrecords  | grep -E '\.tfrecord$' | sort > $tmp2

paste -d '\n' $tmp1 $tmp2 | tr '\n' ' ' | xargs ./record_verifier.py

rm $tmp1 $tmp2

