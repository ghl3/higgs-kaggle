

all: data

data/test.zip:
	echo "Please download the testing data from here: https://www.kaggle.com/c/higgs-boson/data to ./data/test.zip"

data/training.zip:
	echo "Please download the training data from here: https://www.kaggle.com/c/higgs-boson/data to ./data/training.zip"

data/test.csv: data/test.zip
	yes | unzip data/test.zip -d data
	touch data/test.csv

data/training.csv: data/training.zip
	yes | unzip data/training.zip -d data
	touch data/training.csv

data: data/test.csv data/training.csv
	echo "Downloaded and unzipped all data"


image:
	docker build . -t higgs


notebook-server: image
#	docker run -t -i -p 8888:8888 -v $(PWD)/data:/home/ubuntu/data -v $(PWD)/notebooks:/home/ubuntu/notebooks higgs:latest
	docker run -t -i -p 8888:8888 -v $(PWD):/home/ubuntu  higgs:latest
