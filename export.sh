#!/usr/bin/env bash

./build.sh

docker save glas | gzip -c > glas.tar.gz
