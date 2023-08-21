#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# create the required directories
mkdir $SCRIPT_DIR/external
mkdir $SCRIPT_DIR/data
mkdir $SCRIPT_DIR/figures