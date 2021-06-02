#! /usr/bin/bash

shuffler_dir_path="../csc500-utils/steves_utils/ORACLE/"

source_dataset_path="/mnt/wd500GB/CSC500/csc500-super-repo/datasets/all_shuffled_chunk-512/output"
source_dataset_samples_per_chunk="512"
output_batch_size="100"
seed="1337"
num_windowed_examples_per_device="200000"
num_val_examples_per_device="10000"
num_test_examples_per_device="50000"
output_max_file_size_MB="100"
output_window_size="128"

        # "distances_to_filter_on": [
        #     2,
        #     8,
        #     14,
        #     20,
        #     26,
        #     32,
        #     38,
        #     44,
        #     50,
        #     56,
        #     62
        # ],

#for distance in 2 4 8 14 20 26 32 38 44 50 56 62; do
for distance in 8; do
    (
        working_dir="/mnt/wd500GB/CSC500/csc500-super-repo/datasets/automated_windower/windowed_EachDevice-200k_batch-100_stride-20_distances-$distance/"
        mkdir -p $working_dir &&
        cd $shuffler_dir_path &&
        # cat << EOF
        cat << EOF | ./windowed_dataset_shuffler.py -
        {
            "input_shuffled_ds_dir": "$source_dataset_path",
            "input_shuffled_ds_num_samples_per_chunk": $source_dataset_samples_per_chunk,
            "output_batch_size": $output_batch_size,
            "seed": $seed,
            "num_windowed_examples_per_device": $num_windowed_examples_per_device,
            "num_val_examples_per_device": $num_val_examples_per_device,
            "num_test_examples_per_device": $num_test_examples_per_device,
            "output_max_file_size_MB": $output_max_file_size_MB,
            "distances_to_filter_on": [
                $distance
            ],
            "output_window_size": $output_window_size, 
            "working_dir": "$working_dir",
            "stride_length": 20
        }
EOF
    )

done
