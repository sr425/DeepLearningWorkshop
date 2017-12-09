# Deep Learning Workshop Stuttgart 09.12.2017
Some sample and utility code for the Deep Learning Workshop in Stuttgart on 09.12.2017.
Dependencies:
* [Python 3.5/3.6](https://www.python.org/downloads/release/python-363/)
* [Tensorflow (with Slim)](https://www.tensorflow.org/install/)
* Scipy
* Numpy

For Windows get Scipy and Numpy from (https://www.lfd.uci.edu/~gohlke/pythonlibs/) and install with 

```pip install <filename>```

Some information on the different files and folders:
* loaders: some helpers for loading data are given.
* Dataset.py: contains an buffered multithreaded dataset loader for efficient loading of arbitrary datasets.
* Models: contains Googlenet code.
* trainer.py and trainer_onehot.py contain a simple model with all code that is need to train and debug a model.
