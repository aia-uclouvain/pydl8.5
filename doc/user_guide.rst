.. title:: User guide : contents

.. _user_guide:

=====================================================
User guide: Implementation and usage of ODTClassifier
=====================================================

Predictor
---------

Classifier
~~~~~~~~~~

``ODTClassifier`` is a sckit-learn compatible classifier. It can be used as any classifier of
scikit-learn. It implements methods ``fit`` and ``predict``.
.. In addition, it outputs the probabilities of the prediction using the ``predict_proba`` method:
.. Unfortunately, predict_proba is not yet implemented

* at ``fit``, parameters listed in the API documentation <api.html> are learned from ``X`` and ``y``;
* at ``predict``, predictions will be computed using ``X`` using the parameters
  learned during ``fit``. The output corresponds to a list containing the predicted class for each
  sample;
* ``predict_proba`` will give a 2D matrix where each column corresponds to the
  class and each entry will be the probability of the associated class.

In addition, scikit-learn provides a mixin, i.e.
:class:`sklearn.base.ClassifierMixin`, which implements the ``score`` method
which computes the accuracy score of the predictions.

One can import this mixin as::

    >>> from sklearn.base import ClassifierMixin

Therefore, we create a classifier, :class:`ODTClassifier` which inherits
from both :class:`slearn.base.BaseEstimator` and
:class:`sklearn.base.ClassifierMixin`. The method ``fit`` gets ``X`` and ``y``
as input and should return ``self``. It should implement the ``predict``
function which should output the class inferred by the classifier.
``predict_proba`` will output some probabilities instead.

We illustrate that our classifier is working within a scikit-learn pipeline::

    >>> X, y = load_iris(return_X_y=True)
    >>> pipe = make_pipeline(MyOwnClassifier())
    >>> pipe.fit(X, y)  # doctest: +ELLIPSIS
    Pipeline(...)


Then, you can call ``predict`` and ``predict_proba``::

    >>> pipe.predict(X)  # doctest: +ELLIPSIS
    array([...])
    >>> pipe.predict_proba(X)  # doctest: +ELLIPSIS
    array([...])

Since our classifier inherits from :class:`sklearn.base.ClassifierMixin`, we
can compute the accuracy by calling the ``score`` method::

    >>> pipe.score(X, y)  # doctest: +ELLIPSIS
    0...

