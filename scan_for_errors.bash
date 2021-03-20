#! /usr/bin/bash

# There appears to be binaries which do not have a corresponding metadata file. This command finds these cases and compiles an ERRORS file for them.

cat SHA512 | awk '{print }' | xargs -l1 ./metadata_lookup.bash 2>ERRORS
