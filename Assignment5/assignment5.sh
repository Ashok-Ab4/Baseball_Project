mysql -u root -p  -e "create database if not exists baseball;"
mysql -u root -p  baseball < ./baseball.sql
mysql -u root -p  baseball < ./RollingAvgScript.sql