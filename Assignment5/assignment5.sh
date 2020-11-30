#!/bin/sh

sleep 10

if ! mysql -h mariadb-container -uroot -ppassword -e 'use baseball'; then
  mysql -h mariadb-container -uroot -ppassword -e "create database baseball;"
  mysql -h mariadb-container -uroot -ppassword -D baseball < /data/baseball.sql
fi

mysql -h mariadb-container -uroot -ppassword baseball < /scripts/RollingAvgScript.sql

mysql -h mariadb-container -uroot -ppassword baseball -e '
SELECT * FROM RollingBattingAvg;' > /results/RollingAvg.csv
