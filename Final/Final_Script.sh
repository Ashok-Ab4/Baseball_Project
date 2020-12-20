#!/usr/bin/env bash

echo "lets check if baseball exists"
if ! mysql -h mariadb-container -u root  -e 'use baseball'; then
  mysql -h mariadb-container -u root  -e "create database baseball;"
  mysql -h mariadb-container -u root  baseball < ./scripts/baseball.sql
else
  echo "baseball exists"
fi

mysql -h mariadb-container -u root baseball < ./scripts/Finals_FE.sql

#Creating the csv
mysql -h mariadb-container -u root baseball -e '
  SELECT * FROM OutputTable;' > ./scripts/OutputTable.csv

#Calling Python
python3 ./scripts/Model.py



