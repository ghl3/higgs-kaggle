#!/bin/bash -e

docker build . -t higgs
docker run -t -i -v $(PWD):/home/ubuntu higgs:latest
