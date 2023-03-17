![Travis](https://app.travis-ci.com/aglingael/PyDL8.5.svg?branch=master)
![CircleCI](https://circleci.com/gh/aglingael/PyDL8.5/tree/master.svg?style=shield)
![ReadTheDocs](https://readthedocs.org/projects/pydl85/badge/?version=latest)
![Codecov](https://codecov.io/gh/aglingael/PyDL8.5/branch/master/graph/badge.svg?token=UAP32DK54M)
[![Linux - Build and Publish on PyPI](https://github.com/aia-uclouvain/pydl8.5/actions/workflows/Linux-test-and-publish.yml/badge.svg)](https://github.com/aia-uclouvain/pydl8.5/actions/workflows/Linux-test-and-publish.yml)
[![Mac - Build and Publish on PyPI](https://github.com/aia-uclouvain/pydl8.5/actions/workflows/Mac-test-and-publish.yml/badge.svg)](https://github.com/aia-uclouvain/pydl8.5/actions/workflows/Mac-test-and-publish.yml)
[![Windows - Build and Publish on PyPI](https://github.com/aia-uclouvain/pydl8.5/actions/workflows/Windows-test-and-publish.yml/badge.svg)](https://github.com/aia-uclouvain/pydl8.5/actions/workflows/Windows-test-and-publish.yml)


| Authors       | Gaël Aglin, Siegfried Nijssen, Pierre Schaus |
| ----------- |----------------------------------------------|


### Relevant papers
[[DL852020]](https://dial.uclouvain.be/pr/boreal/fr/object/boreal%3A223390/datastream/PDF_01/view) [![DOI](https://img.shields.io/badge/DOI:DL8.5-10.1609%2Faaai.v34i04.5711-blue?logo=adobe-acrobat-reader)](https://doi.org/10.1609/aaai.v34i04.5711)

[[PYDL852020]](https://www.ijcai.org/Proceedings/2020/0750.pdf) [![DOI](https://img.shields.io/badge/DOI:PYDL8.5-10.24963%2Fijcai.2020/750-blue?logo=adobe-acrobat-reader)](https://doi.org/10.24963/ijcai.2020/750)


### Description
**The PyDL8.5 library provides an implementation of DL8.5 algorithm.
Please read the relevant articles referenced below to learn about the
additional features. Please cite these papers if you use the current
library. The documentation will help you get started with PyDL8.5. It is
available [here](https://pydl85.readthedocs.io/en/latest/).**

This project implements an algorithm for inferring optimal binary
decision trees. The core code is written in C++ and this library is
a python wrapper extended by additional features. It is scikit-learn 
compatible and can be used in combination with scikit-learn. As a 
scikit-learn classifier, it implements the methods "fit" and "predict".

Because the search algorithm and the bounds used in DL8.5 are problem-agnostic,
the library can be used to learn any binary decision tree that optimizes
an arbitrary objective function. The only requirement is that the criterion
function should be additive, i.e. the value of the criterion function for
a node should be equal to the sum of the values of the criterion function
for its children. The library provides a default misclassification rate
criterion function, but it can be easily replaced by any other additive
criterion function. As the user-defined criterion function is defined in
Python, the learning process will be slower than the default one. Using this
feature, the library can be used to learn decision trees for many ML tasks
such as regression, classification, clustering, etc. It can solve many problems
such as cost-sensitive classification, multi-class classification, false
positive/negative minimization, feature selection, unfairness
mitigation, complex model explanations, etc. The different criterion functions
can be also easily added to the core C++ code for faster learning.

The core C++ code can be used as a standalone tool. It is available in the
`core` folder. There is a cmake file
to compile the code. The code is also compatible with Windows, Linux and
MacOS. After compilation, the tool can be used in the console by typing
`./dl85 -h` to get the list of available options. There is no dependency
to any other library than the standard C++ library. The only requirement
is a C++ compiler supporting C++17 for Unix systems and C++20 for Windows.
The code is compatible with GCC, Clang and MSVC. The
argument parsing code used originates from the
[argpase](https://github.com/p-ranav/argparse) GitHub project.

The current version of PyDL8.5 is enhanced using some ideas from
[MurTree](https://www.jmlr.org/papers/volume23/20-520/20-520.pdf) paper
and listed in CHANGES.txt. The version of the code used in the AAAI
paper [[DL852020]](https://dial.uclouvain.be/pr/boreal/fr/object/boreal%3A223390/datastream/PDF_01/view) is v0.0.15.

This tool can be installed in two ways:

-   download the source from GitHub and install using the command
    `pip install .` in the root folder.
-   install from `PYPI` using `pip` by running the command `pip install pydl8.5` in the
    console

A C++ compiler supporting C++20 is required to build the code as a
Python library if there is no wheel available for your machine.


### References
| [[DL852020]](https://dial.uclouvain.be/pr/boreal/fr/object/boreal%3A223390/datastream/PDF_01/view) | Aglin, G., Nijssen, S., Schaus, P. | Learning optimal decision trees using caching branch-and-bound search. In AAAI. 2020. |
|----------------| --- | --- |

| [[PYDL852020]](https://www.ijcai.org/Proceedings/2020/0750.pdf) | Aglin, G., Nijssen, S., Schaus, P. | PyDL8.5: a Library for Learning Optimal Decision Trees., In IJCAI. 2020. |
|------------------| --- | --- |
