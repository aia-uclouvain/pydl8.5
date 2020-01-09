|Travis|_ |CircleCI|_ |ReadTheDocs|_ |Codecov|_

.. |Travis| image:: https://travis-ci.org/aglingael/dl8.5.svg?branch=master
.. _Travis: https://travis-ci.org/aglingael/dl8.5

.. |CircleCI| image:: https://circleci.com/gh/aglingael/dl8.5/tree/master.svg?style=svg
.. _CircleCI: https://circleci.com/gh/aglingael/dl8.5/

.. |ReadTheDocs| image:: https://readthedocs.org/projects/dl85/badge/?version=latest
.. _ReadTheDocs: https://dl85.readthedocs.io/en/latest/?badge=latest

.. |Codecov| image:: https://codecov.io/gh/aglingael/dl8.5/branch/master/graph/badge.svg
.. _Codecov: https://codecov.io/gh/aglingael/dl8.5

:Authors:
    Gaël Aglin, Siegfried Nijssen, Pierre Schaus

`Relevant paper <https://dial.uclouvain.be/pr/boreal/fr/object/boreal%3A223390/datastream/PDF_01/view>`_: [DL852020]_

This project implements an algorithm for inferring optimal binary decision trees classifiers.
It is scikit-learn compatible package which provide classifiers and clustering algorithms
and can be used with any scikit-learn functions. As any scikit-learn estimators, you have
to use methods "fit" and "predict".

This tool can be installed by two ways:

* download the sources from github and compile using the command ``python3 setup.py install`` in the root folder
* install from pip by using the command ``pip install dl8.5`` in your console

*Installation from sources ensure you to have up-to-date functionalities when* ``pip`` *method ensure you to have last release.*

The `complete documentation <https://dl85.readthedocs.io/en/latest/?badge=latest>`_ is available at https://dl85.readthedocs.io/en/latest/?badge=latest

.. [DL852020] Aglin, G., Nijssen, S., Schaus, P. Learning optimal decision trees using caching branch-and-bound search. In AAAI. 2020.