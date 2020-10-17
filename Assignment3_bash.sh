#!/bin/bash

#Run the following script to execute the Python script for assignment 3

pip-compile --output-file=requirements.dev.txt requirements.dev.in --upgrade

pip-compile --output-file=requirements.txt requirements.in --upgrade

pip3 install -r requirements.txt

pre-commit install

pre-commit run --all-files

echo "Calling the BDA696 Assignment 3 python Code"
# Run the Assignment file
python ./Assignment_3.py