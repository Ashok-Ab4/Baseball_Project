FROM ubuntu
RUN apt-get update
RUN apt-get install mariadb-client -y

COPY baseball.sql /data/baseball.sql
COPY RollingAvgScript.sql /scripts/RollingAvgScript.sql
COPY assignment5.sh /scripts/assignment5.sh

CMD /scripts/assignment5.sh
