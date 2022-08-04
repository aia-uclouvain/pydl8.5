|Travis|_ |CircleCI|_ |GithubCI|_ |ReadTheDocs|_ |Codecov|_

.. |Travis| image:: https://app.travis-ci.com/aglingael/PyDL8.5.svg?branch=master
.. _Travis: https://app.travis-ci.com/github/aglingael/PyDL8.5

.. |CircleCI| image:: https://circleci.com/gh/aglingael/PyDL8.5/tree/master.svg?style=shield
.. _CircleCI: https://app.circleci.com/pipelines/github/aglingael/PyDL8.5

.. |GithubCI| image:: https://github.com/aia-uclouvain/pydl8.5/actions/workflows/test-and-upload-to-pypi.yml/badge.svg
.. _GithubCI: https://github.com/aia-uclouvain/pydl8.5

.. |ReadTheDocs| image:: https://readthedocs.org/projects/pydl85/badge/?version=latest
.. _ReadTheDocs: https://pydl85.readthedocs.io/en/latest/

.. |Codecov| image:: https://codecov.io/gh/aglingael/PyDL8.5/branch/master/graph/badge.svg?token=UAP32DK54M
.. _Codecov: https://codecov.io/gh/aglingael/PyDL8.5

:Authors:
    Gaël Aglin, Siegfried Nijssen, Pierre Schaus

Relevant paper : `[DL852020] <https://dial.uclouvain.be/pr/boreal/fr/object/boreal%3A223390/datastream/PDF_01/view>`_ `[PYDL852020] <https://www.ijcai.org/Proceedings/2020/0750.pdf>`_

**The PyDL8.5 library provides an implementation of DL8.5 algorithm. Please read the relevant articles referenced below to learn about the additional features.
Please cite these papers if you use the current library. The documentation will help you get started with PyDL8.5.**


This project implements an algorithm for inferring optimal binary decision trees.
It is scikit-learn compatible and can be used in combination with scikit-learn.
As a scikit-learn classifier, it implements the methods "fit" and "predict".

The current version of PyDL8.5 is an optimized one using some ideas from  `MurTree <https://www.jmlr.org/papers/volume23/20-520/20-520.pdf>`_ paper and listed in CHANGES.txt. The version of the code used in the AAAI paper [DL852020] is v0.0.15.

This tool can be installed in two ways:

* download the source from github and install using the command ``python3 setup.py install`` in the root folder
* install from pip by using the command ``pip install pydl8.5`` in the console

The core code provided in C++ can also be used solely without the add-ons provides by the python library.
For this, use the code inside the `core` folder. It is a C++ project than can be compiled using CMake and the CMakeLists.txt file provided.
The argument parsing code used originates from the `argpase <https://github.com/p-ranav/argparse>`_ github project.

**Disclaimer: The compilation of the project has been tested with C++ compilers on the Linux and MacOS operating systems; Windows is not yet supported.**

.. [DL852020] Aglin, G., Nijssen, S., Schaus, P. Learning optimal decision trees using caching branch-and-bound search. In AAAI. 2020.
.. [PYDL852020] Aglin, G., Nijssen, S., Schaus, P. PyDL8.5: a Library for Learning Optimal Decision Trees., In IJCAI. 2020.