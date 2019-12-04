.. title:: User guide : contents

.. _user_guide:

===================================
User guide: Usage of DL85Classifier
===================================

Predictor
---------

Classifier
~~~~~~~~~~

This project implements a class ``DL85Classifier``. 
``DL85Classifier`` is a scikit-learn compatible classifier. It can be used as any classifier of
scikit-learn. It inherits from :class:`sklearn.base.BaseEstimator` and reimplements the methods ``fit`` and ``predict``.

* when the ``fit(X,y)`` method is executed, the attributes listed in the `API documentation <api.html>`_ are learned from ``X`` and ``y``, where ``X`` is a set of Boolean training samples and ``y`` is the  vector of target values;
* when the ``predict(X)`` method is executed, predictions will be computed for the Boolean test samples ``X`` using the attributes
  learned during ``fit``. The output corresponds to a list containing the predicted class for each
  sample.

.. In addition, scikit-learn provides a mixin, i.e. :class:`sklearn.base.ClassifierMixin`, which implements the ``score`` method which computes the accuracy score of the predictions.

.. One can import this mixin as::

..    >>> from sklearn.base import ClassifierMixin
.. The method ``fit`` gets ``X`` and ``y``
.. as input and should return ``self``. It should implement the ``predict``
.. function which should output the class inferred by the classifier.

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

