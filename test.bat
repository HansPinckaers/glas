call .\build.bat

docker volume create glas-output

docker run --rm^
 --memory=4g^
 -v %~dp0\test\:/input/^
 -v glas-output:/output/^
 glas

docker run --rm^
 -v glas-output:/output/^
 python:3.6-slim cat /output/metrics.json | python -m json.tool

docker volume rm glas-output
