#!/usr/bin/env bash
set -e

CONFIG_PATH=$1

# If CONFIG_PATH is not provided, use the default kubeconfig
if [ -z "$CONFIG_PATH" ]; then
    CONFIG_PATH=~/.kube/config
fi

CILANTROPOD=$(kubectl get pods --kubeconfig ${CONFIG_PATH} | awk '/cilantroscheduler/ {print $1;exit}')
kubectl cp $CILANTROPOD:/cilantro/workdirs ./workdirs_eks/ --kubeconfig ${CONFIG_PATH}
# Copy cilantro logs:
LATESTDIR=$(ls -td ./workdirs_eks/*/ | head -1)
kubectl logs $CILANTROPOD --kubeconfig ${CONFIG_PATH} > ${LATESTDIR}cilantroscheduler.log
echo Results are fetched in workdirs_eks