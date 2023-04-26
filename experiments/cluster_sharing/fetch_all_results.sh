#!/usr/bin/env bash
set -e

KUBECONFIGS="./kubeconfig/kc_fin_w2_clus1_1.yaml ./kubeconfig/kc_fin_w2_clus1_2.yaml ./kubeconfig/kc_fin_w2_clus1_3.yaml ./kubeconfig/kc_fin_w2_clus1_4.yaml ./kubeconfig/kc_fin_w2_clus1_5.yaml ./kubeconfig/kc_fin_w2_clus1_6.yaml"
for kc in $KUBECONFIGS; do
  echo Fetching from $kc
  ./starters/fetch_results_kubeconfig.sh $kc
done