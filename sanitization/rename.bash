#! /usr/bin/bash

cat BIN_SHA512 | while read line; do
    sha=$(echo $line | awk '{print $1}')
    name=$(echo $line | awk '{print $2}')

    json=$(./metadata_lookup.bash $sha)

    read -r day transmitter_id transmission_id <<< $(printf '%s\n' "$json" | jq -r '(.day|tostring) + " " + (.transmitter_id|tostring) + " " + (.transmission_id|tostring)')

    echo bin/$name  bin/day-${day}_transmitter-${transmitter_id}_transmission-${transmission_id}.bin
    mv bin/$name  bin/day-${day}_transmitter-${transmitter_id}_transmission-${transmission_id}.bin
done
