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
    supervised.classifiers.Boosting_Model
    predictors.predictor.DL85Predictor
    predictors.predictor.Cache_Type
    predictors.predictor.Wipe_Type
    unsupervised.clustering.DL85Cluster

