#! /bin/bash
for f in $(ls bin); do
    b=$(du -b bin/$f | awk '{print $1}')

    if ! (( $b % 38400 )); then
        echo "            " \"../../csc500-dataset-preprocessor/bin/$f\",
    fi
done
