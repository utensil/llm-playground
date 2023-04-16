#!/bin/bash

cd "$(dirname "$0")"  # change directory to where the script is located

pip3 install -r requirements.txt

export SUDO=${SUDO:-"sudo"}

if [ "$SUDO" = "nosudo" ]; then
    export SUDO=""
fi

$SUDO apt-get update

cat packages.txt | xargs $SUDO apt-get install -y

