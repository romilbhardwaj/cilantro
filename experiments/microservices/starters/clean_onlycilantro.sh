#!/usr/bin/env bash
set -e

CONFIG_PATH=$1

# If CONFIG_PATH is not provided, use the default kubeconfig
if [ -z "$CONFIG_PATH" ]; then
    CONFIG_PATH=~/.kube/config
fi

kubectl delete -f ./starters/hotel-res/cilantro-hr-client.yaml  --grace-period=0 --force --kubeconfig ${CONFIG_PATH}
kubectl delete -f ./starters/cilantro_cfgs/config_cilantro_scheduler_propfair.yaml  --grace-period=0 --force --kubeconfig ${CONFIG_PATH}