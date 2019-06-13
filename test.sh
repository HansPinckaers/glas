#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

./build.sh

docker volume create glas-output

docker run --rm \
        --memory=4g \
        -v $SCRIPTPATH/test/:/input/ \
        -v glas-output:/output/ \
        glas

docker run --rm \
        -v glas-output:/output/ \
        python:3.6-slim cat /output/metrics.json | python -m json.tool

docker volume rm glas-output
