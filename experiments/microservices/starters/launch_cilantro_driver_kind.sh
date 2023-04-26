#!/usr/bin/env bash
# Runs the Cilantro scheduler and client for the microservices experiment
# Usage:
#   ./starters/launch_cilantro_driver.sh <CONFIG_PATH> <POLICY>
# Example:
#   ./starters/launch_cilantro_driver.sh ~/.kube/config ucbopt

set -e

CONFIG_PATH=$1
POLICY=$2

# Assert that both CONFIG_PATH and POLICY are provided
if [ -z "${CONFIG_PATH}" ]; then
    echo "CONFIG_PATH not provided. Exiting. Consider passing ~/.kube/config."
    exit 1
fi

if [ -z "${POLICY}" ]; then
    echo "POLICY not provided. Exiting. Possible options are: propfair, ucbopt, msile, msevoopt."
    exit 1
fi

kubectl apply -f ./starters/hotel-res/auth_default_user.yaml --kubeconfig ${CONFIG_PATH}
kubectl apply -f ./starters/hotel-res/dashboard.yaml --kubeconfig ${CONFIG_PATH}
kubectl apply -f ./starters/hotel-res/cilantro-hr-client.yaml --kubeconfig ${CONFIG_PATH}
kubectl apply -f ./starters/cilantro_cfgs/kind/config_cilantro_scheduler_${POLICY}.yaml --kubeconfig ${CONFIG_PATH}

