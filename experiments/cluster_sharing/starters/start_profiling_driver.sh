#!/usr/bin/env bash
set -e

CONFIG_PATH=$1

# If CONFIG_PATH is not provided, use the default kubeconfig
if [ -z "$CONFIG_PATH" ]; then
    CONFIG_PATH=~/.kube/config
fi

kubectl apply -f ./starters/auth_default_user.yaml  --kubeconfig ${CONFIG_PATH}
kubectl apply -f ./starters/config_profiling_driver_eks.yaml  --kubeconfig ${CONFIG_PATH}
sleep 10
kubectl cp $(kubectl get pods  --kubeconfig ${CONFIG_PATH} | awk '/cilantroscheduler/ {print $1;exit}'):/cilantro/workdirs ./workdirs_profiling_eks/ --kubeconfig ${CONFIG_PATH}
sleep 10
./starters/fetch_results_clus1_1.sh
