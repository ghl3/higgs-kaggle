FROM ubuntu:16.04
RUN    apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y python2.7-dev python-matplotlib python-pip build-essential \
    && pip install --upgrade pip

RUN pip install jupyter scipy pandas sklearn xgboost

EXPOSE 8888:8888

RUN mkdir -p /home/ubuntu

WORKDIR /home/ubuntu

VOLUME /home/ubuntu/data /home/ubuntu/notebooks

COPY *.py ./

CMD jupyter notebook --ip 0.0.0.0 --no-browser --allow-root