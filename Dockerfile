FROM ubuntu:16.04
RUN    apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y python2.7-dev python-matplotlib python-pip build-essential \
    && pip install --upgrade pip

RUN pip install --upgrade ipython matplotlib numpy jupyter scipy pandas sklearn xgboost seaborn tensorflow keras

EXPOSE 8888:8888

VOLUME /home/ubuntu

WORKDIR /home/ubuntu

COPY *.py Makefile ./

CMD make