#!/usr/bin/env bash

python logparse_tester.py --log-folder-path ../data_sources/log_parsers/example_logs/ycsb \
       --log-extension "*.log" --slo-latency 10000
