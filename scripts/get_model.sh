#!/bin/sh
# Download Granite 3.1 1B-A400M (Apache 2.0, 2.7 GB) and convert it. Needs about 5.4 GB of disk.
set -e
cd "$(dirname "$0")/.."
BASE=https://huggingface.co/ibm-granite/granite-3.1-1b-a400m-instruct/resolve/main
mkdir -p models/src
for f in config.json tokenizer.json model.safetensors; do
    curl -fL -C - -o "models/src/$f" "$BASE/$f"
done
echo "ac02591061f1344027a7e7b11dbb4143f75f166c47dc09b742f5de3ab1dde1d1  models/src/model.safetensors" | sha256sum -c
python -m expertstream.convert models/src models/granite
cp bench/hot.json models/granite/hot.json
