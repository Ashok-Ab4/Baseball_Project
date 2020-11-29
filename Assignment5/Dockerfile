FROM ubuntu
RUN apt-get update
RUN apt-get install mariadb-client -y

WORKDIR ./Assignment5
COPY baseball.sql /baseball.sql
COPY RollingAvgScript.sql /RollingAvgScript.sql
COPY assignment5.sh /assignment5.sh

RUN chmod +x assignment5.sh

CMD assignment5.sh
