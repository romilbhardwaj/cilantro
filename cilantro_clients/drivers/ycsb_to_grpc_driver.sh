#!/usr/bin/env bash
APP_NAME="cassandra-server"
IP="localhost"
PORT=10000

python ycsb_to_grpc_driver.py --log-folder-path /mnt/d/Romil/Berkeley/Research/cilantro/cilantro/cilantro_clients/data_sources/log_parsers/example_logs/ycsb \
                              --grpc-client-id ${APP_NAME} \
                              --grpc-ip ${IP} \
                              --grpc-port ${PORT} \
                              --poll-frequency 5 \
                              --app-name ${APP_NAME}