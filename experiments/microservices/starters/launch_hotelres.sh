#!/usr/bin/env bash
set -e

# Launches the hotel reservation core microservices (19). Does not launch the client.

CONFIG_PATH=$1

# If CONFIG_PATH is not provided, use the default kubeconfig
if [ -z "$CONFIG_PATH" ]; then
    CONFIG_PATH=~/.kube/config
fi

WORKLOAD_K8S_PATH=./starters/hotel-res/hotel-res-core/
kubectl apply -Rf ${WORKLOAD_K8S_PATH} --kubeconfig ${CONFIG_PATH}