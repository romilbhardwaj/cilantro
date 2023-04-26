#!/usr/bin/env bash
KCTL_ARGS=$1
echo "============= HotelReservationPrePull ==============="
echo Prepulling HotelReservation image.
kubectl create -f prepull_hotelres.yaml $KCTL_ARGS
echo Prepull running now. Should complete soon. Note that this needs to be run only once after the cluster has been created with eksctl.
