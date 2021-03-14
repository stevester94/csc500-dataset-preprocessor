set -eou pipefail

metadata_base_path="metadata"

metadata_file_path=$(grep -nrl $1 $metadata_base_path)

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
