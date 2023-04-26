#!/usr/bin/env bash
# Runs the Cilantro scheduler and client for the cluster sharing experiment on kind cluster
# Usage:
#   ./starters/launch_cilantro_driver.sh <CONFIG_PATH> <POLICY>
# Example:
#   ./starters/launch_cilantro_driver.sh ~/.kube/config propfair

set -e

CONFIG_PATH=$1
POLICY=$2

# Assert that both CONFIG_PATH and POLICY are provided
if [ -z "${CONFIG_PATH}" ]; then
    echo "CONFIG_PATH not provided. Exiting. Consider passing ~/.kube/config."
    exit 1
fi

if [ -z "${POLICY}" ]; then
    echo "POLICY not provided. Exiting. Possible options are: propfair, mmf, mmflearn, utilwelforacle, utilwelflearn, evoutil, egalwelforacle, egalwelflearn, evoegal, greedyegal, minerva, ernest, quasar, parties, multincadddec."
    exit 1
fi

# Assume tainting has already been done
kubectl apply -f ./starters/auth_default_user.yaml --kubeconfig ${CONFIG_PATH}
kubectl apply -f ./starters/dashboard.yaml --kubeconfig ${CONFIG_PATH}
kubectl apply -f ./starters/cilantro_cfgs/kind/config_cilantro_scheduler_${POLICY}.yaml --kubeconfig ${CONFIG_PATH}

