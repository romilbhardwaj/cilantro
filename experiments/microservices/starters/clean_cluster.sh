#!/usr/bin/env bash
set -e

CONFIG_PATH=$1

# If CONFIG_PATH is not provided, use the default kubeconfig
if [ -z "$CONFIG_PATH" ]; then
    CONFIG_PATH=~/.kube/config
fi

kubectl delete jobs,daemonsets,replicasets,services,deployments,pods,rc,statefulset --all --grace-period=0 --force --kubeconfig ${CONFIG_PATH}
sleep 0.2
kubectl delete jobs,daemonsets,replicasets,services,deployments,pods,rc,statefulset --all --grace-period=0 --force --kubeconfig ${CONFIG_PATH}
sleep 0.2
kubectl delete jobs,daemonsets,replicasets,services,deployments,pods,rc,statefulset --all --grace-period=0 --force --kubeconfig ${CONFIG_PATH}
