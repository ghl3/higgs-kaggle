

all: predictions/predictions.csv

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

data/training_processed.csv: process_data.py features.py data/training.csv data/test.csv
	python process_data.py

predictions/predictions.csv: predict.py data/training_processed.csv
	python predict.py
