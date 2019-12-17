.. DL8.5 documentation master file, created by
   sphinx-quickstart on Sat Nov  9 16:12:43 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to DL8.5's documentation!
=================================

**Decision Trees (DTs)** are a supervised learning method used for *classification* and
*regression*. The goal is to create a model that predicts the value of a target variable by learning
simple decision rules inferred from data.

Although traditional algorithms for learning decision trees provide good results in general, well-known algorithms such as CART and C4.5 do not calculate trees that are necessarily optimal.

This repository contains an implemetation of DL8.5, an algorithm for finding optimal decision trees under formal requirements on the accuracy, support and depth of the decision trees to be found. Details about this algorithm can be found in :cite:`dl852020`. The key idea underlying this new approach
is the use of a cache of itemsets in combination with branch-and-bound search; this new type of cache also stores
results for parts of the search space that have been traversed partially.

This implementation is scikit-learn compatible and can be used in combination with scikit-learn. 

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

This is the API documentation of DL8.5.

`Examples <auto_examples/index.html>`_
--------------------------------------

A set of examples. It complements the `User Guide <user_guide.html>`_.

.. rubric:: References
.. bibliography:: references.bib
