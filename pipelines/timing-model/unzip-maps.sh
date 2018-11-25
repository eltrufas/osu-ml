#!/usr/bin/env bash
set -xe

pushd maps
for filename in ./*.rar; do
	unrar e "$filename"
done
popd

python scripts/unzip_maps.py

