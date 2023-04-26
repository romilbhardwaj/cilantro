#!/usr/bin/env bash
APP_NAME="cray"
IP="localhost"
PORT=10000

python cray_to_grpc_driver.py --log-folder-path ../data_sources/log_parsers/example_logs/cray \
                              --grpc-client-id ${APP_NAME} \
                              --grpc-ip ${IP} \
                              --grpc-port ${PORT} \
                              --poll-frequency 5 \
                              --slo-type latency \
                              --slo-latency 4.2

# python cray_to_grpc_driver.py --log-folder-path ../data_sources/log_parsers/example_logs/cray \
#                               --grpc-client-id ${APP_NAME} \
#                               --grpc-ip ${IP} \
#                               --grpc-port ${PORT} \
#                               --poll-frequency 5 \
#                               --slo-type throughput
