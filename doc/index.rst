.. PyDL8.5 documentation master file, created by
   sphinx-quickstart on Mon Sep 20 13:26:24 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyDL8.5's documentation!
===================================

**Decision Trees (DTs)** are machine learning models used for *classification* and
other prediction tasks. They perform prediction by means of
simple decision rules inferred from data.

Traditional algorithms for learning decision trees, such as CART and C4.5, are
heuristic in nature. However, as a result, the trees that are learned by these
algorithms may sometimes be more complex than necessary, and hence less interpretable.

This repository contains an implementation of DL8.5, an algorithm for finding optimal decision trees under formal requirements on the accuracy, support and depth of the decision trees to be found. Details about this algorithm can be found in :cite:`dl852020` and :cite:`pydl852020`. The key idea underlying this algorithm
is the use of a cache of itemsets in combination with branch-and-bound search; this new type of cache also stores
results for parts of the search space that have been traversed partially. An experimental comparison with other methods in :cite:`dl852020` shows that DL8.5's performance is much better than that of competing methods.

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

This is the API documentation of PyDL8.5.

`Examples <auto_examples/index.html>`_
--------------------------------------

These examples illustrate further how DL8.5 can be used;
for more detailed information, please consult   the `User Guide <user_guide.html>`_.

.. rubric:: References
.. bibliography:: references.bib
