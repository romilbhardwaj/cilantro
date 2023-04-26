#!/usr/bin/env bash
set -e

CONFIG_PATH=$1

# If CONFIG_PATH is not provided, use the default kubeconfig
if [ -z "$CONFIG_PATH" ]; then
    CONFIG_PATH=~/.kube/config
fi

./starters/launch_hotelres.sh ${CONFIG_PATH}
./starters/launch_cilantro_driver.sh ${CONFIG_PATH}
