#!/usr/bin/env bash
APP_NAME="testapp"
IP="localhost"
PORT=10000

python timeseries_to_grpc_driver.py --csv-file example_timeseries.csv --roll-over --client-id ${APP_NAME} --grpc-ip ${IP} --grpc-port ${PORT} --poll-frequency 1