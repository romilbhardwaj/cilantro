#!/usr/bin/env bash
KCTL_ARGS=$1
echo "============= CrayPrePull ==============="
echo Prepulling cray image.
kubectl create -f prepull_cray.yaml $KCTL_ARGS
echo Prepull running now. Should complete soon. Note that this needs to be run only once after the cluster has been created with eksctl.