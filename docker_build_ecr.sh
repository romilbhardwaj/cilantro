#!/usr/bin/env bash
set -e

docker build . -t public.ecr.aws/cilantro/cilantro:latest
docker push public.ecr.aws/cilantro/cilantro:latest
