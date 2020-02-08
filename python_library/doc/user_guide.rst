.. title:: User guide : contents

.. _user_guide:

======================
User guide: Using DL85
======================

Optimal Decision Trees
----------------------

This project implements the DL8.5 algorithm for learning optimal binary decision trees. 
Examples of decision trees are classification trees and regression trees. 
Classification trees are predictors in which the predictions correspond to class labels; 
regression trees are predictors in which the predictions are numerical.

Decision trees are traditionally learned using heuristic algorithms, such as CART and C4.5.
However, due to their heuristic nature, the trees learned using these algorithms can be larger than 
necessary; this makes the resulting trees less interpretable. Trees found by DL8.5 are optimal on training data,
that is, no better tree can be found under user-specified constraints that aim to make the resulting
trees more interpretable.

Please note that trees that perform well on training data, may not always perform good on test data. To avoid 
problems with overfitting, it is recommended to run DL8.5 using carefully chosen constraints, as specified below. 

Classifier
~~~~~~~~~~

Decision tree classifiers are learned using the class ``DL85Classifier``. 
``DL85Classifier`` is a scikit-learn compatible classifier and can be used as a scikit-learn
classifier. It inherits from :class:`sklearn.base.BaseEstimator` and reimplements the methods ``fit`` and ``predict``.

* when the ``fit(X,y)`` method is executed, an optimal decision tree classifier is learned from ``X`` and ``y``, where ``X`` is a set of Boolean training samples and ``y`` is the  vector of target values; the resulting tree is stored in the ``DL85Classifier`` object. For more information on how the results of the learning algorithm are stored, please check the 
 `API documentation <api.html>`_.
* when the ``predict(X)`` method is executed, predictions will be computed for the Boolean test samples ``X`` using the tree
  learned during the execution of ``fit``. The output corresponds to a list of predicted classes for all the
  samples.

Parameters of the learning process need to be specified during the construction of the ``DL85Classifier`` object. 
The complete list of parameters can be found in the `API documentation <api.html>`_. We highly recommend to
specify the parameters of the following constraints, as their default values are not useful in many cases:

* ``max_depth``, which specifies the maximum depth of the tree to be learned; low values will ensure that more interpretable trees are found; high values may lead to better training set accuracy, but can also lead to overfitting;
* ``min_sup``, which specifies the minimum number of examples in the training data that every leaf should cover; low values may lead to predictions that are based on too small amounts of data.

Other parameters that may be useful to tune are:

* ``time_limit``, which indicates the maximum amount of time the algorithm is allowed to run; the algorithm will be interrupted when the runtime is exceeded, and the best tree found within the allocated time will be returned. The default value is ``0``, in which case no limit on runtime is imposed.
* ``max_error``, which will direct the search algorithm to only find trees with an error lower than ``max_error``. For instance, 
if a decision tree has already been found using another algorithm (such as a heuristic algorithm), specifying this parameter could direct DL8.5 to only 
find trees that are better than the tree found using this other algorithm.



.. In addition, scikit-learn provides a mixin, i.e. :class:`sklearn.base.ClassifierMixin`, which implements the ``score`` method which computes the accuracy score of the predictions.

.. One can import this mixin as::

..    >>> from sklearn.base import ClassifierMixin
.. The method ``fit`` gets ``X`` and ``y``
.. as input and should return ``self``. It should implement the ``predict``
.. function which should output the class inferred by the classifier.

The following code illustrates how to use DL8.5 in its most basic setting::

    import numpy as np
    from sklearn.model_selection import train_test_split
    from dl85 import DL85Classifier 

    dataset = np.genfromtext("anneal.txt", delimiter=" ")
    X = dataset[:, 1:]
    y = dataset[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

    clf = DL85Classifier(max_depth=3, min_sup=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

In this example, we use numpy to read a dataset and scikit-learn to split this dataset in training and test data.
Subsequently, the DL85Classifier is initialized, where the recommended parameters are specified; a decision tree is learned 
from the training data, which is applied on the test data.

The following example illustrates how to use our classifier within a scikit-learn pipeline to learn an optimal decision tree classifier::

    >>> dataset = np.genfromtxt("binary_dataset.txt", delimiter=' ')
    >>> X = dataset[:, 1:]
    >>> y = dataset[:, 0]
    >>> pipe = make_pipeline(DL85Classifier())
    >>> pipe.fit(X, y)  # doctest: +ELLIPSIS
    Pipeline(...)


Then, you can call ``predict`` to classify these examples::

    >>> pipe.predict(X)  # doctest: +ELLIPSIS
    array([...])



:class:`DL85Classifier` also inherits from 
:class:`sklearn.base.ClassifierMixin`. This allows the use of the ``score`` method to calculate 
the accuracy of the classifier::

    >>> pipe.score(X, y)  # doctest: +ELLIPSIS
    0...

Other predictors
~~~~~~~~~~~~~~~~

DL8.5 provides an interface that allows other trees than classification trees to be learned, and other scoring functions 
than training set error to be used. In this case, the user has to implement a new scoring function 
in Python. Examples of the use of this interface 
can be found on the page of examples for predictive clustering. 

Note 1: As regression trees are a special case
of predictive clustering trees, this interface can also be used for regression.

Note 2: While we decided to make this preliminary interface publicly available already, 
it may still change in the future 
to make it easier to use. For this reason, the documentation of this interface is currently still short.


