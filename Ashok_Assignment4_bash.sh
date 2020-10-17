#!/bin/bash

#Updating all the requirement files and installing required packages

pip-compile --output-file=requirements.dev.txt requirements.dev.in --upgrade

pip3 install -r requirements.dev.txt

pip-compile --output-file=requirements.txt requirements.in --upgrade

pip3 install -r requirements.txt

pre-commit install

pre-commit run --all-files

read -p "input a csv file name" input_df_filename
read -p "enter name of response variable" response

echo "running the code"

python ./Assignment_4_FE.py $input_df_filename $response