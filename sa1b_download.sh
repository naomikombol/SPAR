#!/usr/bin/env bash

set -euo pipefail

# CONFIG
LINKS_FILE="./datasets/sa1b_links.txt"
OUTPUT_DIR="YOUR_RAW_IMAGE_OUTPUT_DIRECTORY"
MAX_INDEX=25000

# Choose which indices to download
INDICES=(0 1 2)

mkdir -p "$OUTPUT_DIR"

format_name() {
    printf "sa_%06d.tar" "$1"
}

for idx in "${INDICES[@]}"; do
    FILE_NAME=$(format_name "$idx")

    echo "Processing $FILE_NAME..."

    URL=$(awk -v name="$FILE_NAME" '$1 == name {print $2}' "$LINKS_FILE")

    if [[ -z "$URL" ]]; then
        echo "Warning: No URL found for $FILE_NAME"
        continue
    fi

    FILE_PATH="$OUTPUT_DIR/$FILE_NAME"

    echo "Downloading..."
    wget -c "$URL" -O "$FILE_PATH"

    echo "Extracting (excluding JSON)..."
    tar -xf "$FILE_PATH" -C "$OUTPUT_DIR" --exclude='*.json'

    echo "Removing archive..."
    rm -f "$FILE_PATH"

    echo "Filtering images > sa_$(printf "%06d" $MAX_INDEX)..."
    find "$OUTPUT_DIR" -type f -name "sa_*.jpg" | while read -r file; do
        base=$(basename "$file")
        num=${base#sa_}
        num=${num%%.*}

        num=$((10#$num))

        if (( num > MAX_INDEX )); then
            rm -f "$file"
        fi
    done

    echo "Done with $FILE_NAME"
    echo "-----------------------------"
done

echo "All done."
