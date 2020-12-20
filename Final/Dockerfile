FROM ubuntu
RUN mkdir /scripts

COPY ./baseball.sql /scripts/baseball.sql
COPY ./Finals_FE.sql /scripts/Finals_FE.sql
COPY ./Final_Script.sh /scripts/Final_Script.sh
COPY ./cat_correlation.py /scripts/cat_correlation.py
COPY ./Assignment_4_FE.py /scripts/Assignment_4_FE.py
COPY ./Midterm_FE_Ashok.py /scripts/Midterm_FE_Ashok.py
COPY ./Model.py /scripts/Model.py
RUN chown 1000:1000 /scripts

RUN apt-get update \
  && apt-get install --no-install-recommends --yes \
     build-essential \
     libmysqlclient-dev \
     mysql-client \
     python3 \
     python3-pip \
     python3-dev \
     python3-pymysql \
  && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /scripts/requirements.txt
RUN pip3 install --compile --no-cache-dir -r /scripts/requirements.txt

RUN chmod +x /scripts/Final_Script.sh
CMD ./scripts/Final_Script.sh
