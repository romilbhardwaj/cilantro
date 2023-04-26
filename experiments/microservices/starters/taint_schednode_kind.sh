#!/usr/bin/env bash
# Adds the scheduler label to the main node for kind clusters

KCTL_ARGS=$1
echo $KCTL_ARGS

echo "============= TaintingScript for Kind clusters ==============="
echo "=========== Step 1 - Clean up ==============="
# Clear taint from all nodes first
echo Removing taints from all nodes. Ignore any not found errors.
NODES_LIST=$(kubectl get nodes $KCTL_ARGS | tail -n +2 | awk '{print $1}')
kubectl taint nodes $NODES_LIST dedicated=scheduler:NoSchedule- $KCTL_ARGS

echo Removing scheduler labels from all nodes. Ignore any not found errors.
kubectl label node $NODES_LIST dedicated- $KCTL_ARGS

echo "=========== Step 2 - Add taints and label to scheduler ==============="
# Pick a scheduler node:
NODE_TO_TAINT=$(kubectl get nodes $KCTL_ARGS | sed -n '2 p' | cut -d" " -f1)

# Taint and label
echo Adding labels to scheduler node
kubectl label nodes $NODE_TO_TAINT dedicated=scheduler $KCTL_ARGS

echo Tainting done. Make sure to specify dedicated=scheduler as a toleration and nodeSelector in the podspec.