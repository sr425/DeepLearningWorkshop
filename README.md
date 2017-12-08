# Deep Learning Workshop Stuttgart 09.12.2017
Some sample and utility code for the Deep Learning Workshop in Stuttgart on 09.12.2017.
Dependencies:
* Python 3.5
* Tensorflow (with Slim)
* Scipy
* Numpy

Some information on the different files and folders:
* loaders: some helpers for loading data are given.
* Dataset.py: contains an buffered multithreaded dataset loader for efficient loading of arbitrary datasets.
* Models: contains Googlenet code.
* trainer.py and trainer_onehot.py contain a simple model with all code that is need to train and debug a model.