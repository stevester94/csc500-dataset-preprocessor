# /usr/bin/bash

# Find move all the error cases to their own dir
mkdir  error_bin
cat ERRORS | awk '{print }' | xargs -l1 -I% grep % SHA512 | awk '{print }' | xargs -l1 -I% mv bin/% error_bin/%
