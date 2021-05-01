#! /usr/bin/bash

set -eou pipefail
in_path=$1
out_path=$2

mkdir -p $out_path

parallelization=5
count=0

echo set -eou pipefail
for f_in in $(ls $in_path/*bin); do
    base_name=$(basename $f_in)
    base_name=${base_name%.bin}

    out_name="${base_name}_batch-1_shuffled.ds"

    echo ./shuffle_vanilla_binary_to_dataset.py $f_in $out_path/$out_name

    if ! (( $count % $parallelization )); then
        echo "wait"
    fi

    (( count = count + 1 ))
done