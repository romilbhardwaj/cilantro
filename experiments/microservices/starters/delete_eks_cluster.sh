#!/usr/bin/env bash
set -e
eksctl delete cluster -f ./starters/eks_cluster_cfg.yaml