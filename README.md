
# Higgs Modeling


This repository contains:
- Definitions of new features derived from the raw input data
- Scripts to process the input data, split it into cross validation set, add new features, and save the results
- Scripts to read the processed data, fit a model, run that model on the test data, and write the results of the test data to a directory
- A jupyter notebook which walks through the model design and building process, showing different iterations and why the final version was chosen
- Docker files so that the analysis can be run in an environment that has all the necessary dependencies

To view the analysis, see <a href='./notebooks/modeling.ipynb'>notebooks/modeling.ipynb</a>


To run a notebook server from within a Docker container (that can be accessed from outside the container), do:

    > run_notebook.sh
    
    
To process features, build a model, and make predictions on the test set (from within a Docker container), do:

    > ./run_predictions.sh
