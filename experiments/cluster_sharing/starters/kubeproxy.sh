#!/usr/bin/env bash
set -e

CONFIG_PATH=$1

# If CONFIG_PATH is not provided, use the default kubeconfig
if [ -z "$CONFIG_PATH" ]; then
    CONFIG_PATH=~/.kube/config
fi

echo Running kubeproxy. Keep this terminal open and visit the dashboard at http://localhost:8001/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard:/proxy/#/persistentvolumeclaim?namespace=_all
kubectl proxy --kubeconfig ${CONFIG_PATH}