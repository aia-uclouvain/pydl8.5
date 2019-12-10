.. DL8.5 documentation master file, created by
   sphinx-quickstart on Sat Nov  9 16:12:43 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to DL8.5's documentation!
=================================

**Decision Trees (DTs)** are a non-parametric supervised learning method used for *classification* and
*regression*. The goal is to create a model that predicts the value of a target variable by learning
simple decision rules inferred from the data features.

However DTs provide good results in general, the ways used by weel-known algorithms to infer them are sub-optimal.
For many reasons, it arrives sometimes when we want DTs very accurate as possible depending on a error function.
Several recent publications have studied the use of Mixed Integer Programming (MIP) for finding an optimal decision
tree, that is, the best decision tree under formal requirements on accuracy, fairness or interpretability of the
predictive model. These publications used MIP to deal with the hard computational challenge of finding such trees.
In this work, we implement a new efficient algorithm named **DL8.5**, for finding optimal decision trees, based on
the use of itemset mining techniques. We show through :cite:`dl852020` that this new algorithm outperforms earlier approaches with several orders
of magnitude, for both numerical and discrete data, and is generic as well. The key idea underlying this new approach
is the use of a cache of itemsets in combination with branch-and-bound search; this new type of cache also stores
results for parts of the search space that have been traversed partially.

Thus, this project implements **DL8.5** for inferring binary optimal decision tree classifiers.
It provides a scikit-learn compatible classifier which can be used with any scikit-learn functions.

.. As any scikit-learn classifier, you have to use methods "fit" and "predict".

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   user_guide

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Documentation

   api

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Tutorial - Examples

   auto_examples/index

`Getting started: User guide <user_guide.html>`_
------------------------------------------------

This is a small tutorial that explains how to use DL8.5. 

`API Documentation <api.html>`_
-------------------------------

Found here the API documentation of DL8.5.

`Examples <auto_examples/index.html>`_
--------------------------------------

A set of examples. It complements the `User Guide <user_guide.html>`_.

.. rubric:: References
.. bibliography:: references.bib