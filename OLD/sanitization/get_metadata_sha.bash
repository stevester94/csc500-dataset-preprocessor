#! /usr/bin/bash

metadata_base_path="metadata"

SHA512_file_path=SHA512

for metadata_file_path in $(ls $metadata_base_path); do
    sha512=$(jq -r '._metadata.global["core:sha512"]' $metadata_base_path/$metadata_file_path)

    echo  $sha512 $metadata_file_path
done
