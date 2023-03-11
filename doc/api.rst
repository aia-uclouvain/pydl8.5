####################
PyDL8.5 API
####################

This project implements the class ``DL85Classifier`` for learning optimal classification trees using the DL8.5 algorithm. Moreover, it provides a ``DL85Predictor`` class that 
provides an interface for the implementation of other decision tree learning tasks.
The ``DL85Cluster`` class supports a form of predictive clustering.

The documentation for these classes is given below.

.. currentmodule:: pydl85

Predictors
==========

.. autosummary::
   :toctree: generated/
   :template: class.rst

    supervised.classifiers.DL85Classifier
    supervised.classifiers.DL85Booster
    predictors.predictor.DL85Predictor
    unsupervised.clustering.DL85Cluster

