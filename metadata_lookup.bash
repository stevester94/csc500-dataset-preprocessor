#! /usr/bin/bash

if [[ -z $1 ]]; then
    echo "Usage: <SHA512 of the binary you want to find the metadata for>"
    exit 1
fi
    

metadata_base_path="metadata"

metadata_file_path=$(grep -nrl $1 $metadata_base_path)

if [[ $? != 0 ]]; then
    echo "NOTFOUND $1" 1>&2
    exit 1
fi

if [[ $(echo $metadata_file_path | tr ' ' '\n' | wc -l) != 1 ]]; then
    echo "MULTIPLE $1" 1>&2
    exit 1
fi

sha512=$(jq -r '._metadata.global["core:sha512"]' $metadata_file_path)
transmitter_id=$(jq -r '._metadata.annotations[0]["wines:transmitter"].ID["Transmitter ID"]' $metadata_file_path)
transmission_id=$(jq -r '._metadata.annotations[0]["wines:transmitter"].ID["Transmission ID"]' $metadata_file_path)
day=$(jq -r '.data_file' $metadata_file_path | grep -o -E 'Day_[0-9]+' | grep -o -E '[0-9]+')

sample_count=$(jq -r '._metadata.annotations[0]["core:sample_count"]' $metadata_file_path)


cat << EOF
{
    "sha512": "$sha512",
    "transmitter_id": "$transmitter_id",
    "transmission_id": "$transmission_id"
    "day": "$day",
    "sample_count": "$sample_count",
}
EOF
