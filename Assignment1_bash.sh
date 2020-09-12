#!/bin/bash

#Updating all the requirement files and installing required packages

pip-compile --output-file=requirements.dev.txt requirements.dev.in --upgrade

pip3 install -r requirements.dev.txt

pip-compile --output-file=requirements.txt requirements.in --upgrade

pip3 install -r requirements.txt

pre-commit install

pre-commit run --all-files

echo "running the code"

python ./Assignment1.py