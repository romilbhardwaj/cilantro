#!/usr/bin/env bash
python -m grpc_tools.protoc -I./protobufs --python_out=./protogen --grpc_python_out=./protogen ./protobufs/utility_update.proto