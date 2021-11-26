from setuptools import find_packages, setup
from setuptools import Extension
from Cython.Build import cythonize
import platform
import codecs
import os.path


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r', encoding='utf-8-sig') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


DISTNAME = 'pydl8.5'
DESCRIPTION = 'A package to build an optimal binary decision tree classifier.'
LONG_DESCRIPTION = read('README.rst')
AUTHORS = 'Gael Aglin, Siegfried Nijssen, Pierre Schaus'
AUTHORS_EMAIL = 'aglingael@gmail.com, siegfried.nijssen@gmail.com, pschaus@gmail.com'
URL = 'https://github.com/aglingael/PyDL8.5'
LICENSE = 'LICENSE.txt'
DOWNLOAD_URL = 'https://github.com/aglingael/PyDL8.5'
VERSION = get_version("dl85/_version.py")
INSTALL_REQUIRES = ['setuptools', 'cython', 'numpy', 'scikit-learn', 'gurobipy', 'cvxpy']
SETUP_REQUIRES = ['setuptools', 'cython', 'numpy', 'scikit-learn', 'gurobipy', 'cvxpy']
KEYWORDS = ['decision trees', 'discrete optimization', 'classification']
CLASSIFIERS = ['Programming Language :: Python :: 3',
               'License :: OSI Approved :: MIT License',
               'Operating System :: OS Independent',
               'Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering']
EXTRAS_REQUIRE = {
    'tests': [
        'pytest',
        'pytest-cov'],
    'docs': [
        'sphinx',
        'sphinx-gallery',
        'sphinx_rtd_theme',
        'numpydoc',
        'sphinxcontrib',
        'matplotlib'
    ]
}
PROJECT_URLS = {
    "Source on github": "https://github.com/aglingael/dl8.5",
    "Documentation": "https://dl85.readthedocs.io/en/latest/?badge=latest",
}

EXTENSION_NAME = 'dl85Optimizer'
EXTENSION_LANGUAGE = 'c++'
EXTENSION_SOURCE_FILES = ['cython_extension/dl85Optimizer.pyx',
                          'cython_extension/error_function.pyx',
                          'core/src/cache.cpp',
                          'core/src/cache_hash.cpp',
                          'core/src/cache_hash_cover.cpp',
                          'core/src/cache_trie.cpp',
                          'core/src/dataManager.cpp',
                          'core/src/depthTwoComputer.cpp',
                          'core/src/dl85.cpp',
                          'core/src/globals.cpp',
                          'core/src/nodeDataManager.cpp',
                          'core/src/nodeDataManagerFreq.cpp',
                          'core/src/rCover.cpp',
                          'core/src/rCoverFreq.cpp',
                          'core/src/rCoverWeight.cpp',
                          'core/src/search_base.cpp',
                          'core/src/search_cache.cpp',
                          'core/src/search_nocache.cpp',
                          'core/src/search_hash_cover.cpp',
                          'core/src/solution.cpp',
                          'core/src/solutionFreq.cpp']
EXTENSION_INCLUDE_DIR = ['core/src', 'cython_extension']
EXTENSION_BUILD_ARGS = ['-std=c++17', '-DCYTHON_PEP489_MULTI_PHASE_INIT=0']
if platform.system() == 'Darwin':
    EXTENSION_BUILD_ARGS.append('-mmacosx-version-min=10.12')

dl85_extension = Extension(
    name=EXTENSION_NAME,
    language=EXTENSION_LANGUAGE,
    sources=EXTENSION_SOURCE_FILES,
    include_dirs=EXTENSION_INCLUDE_DIR,  # path for headers
    extra_compile_args=EXTENSION_BUILD_ARGS,
    extra_link_args=EXTENSION_BUILD_ARGS
)

setup(
    name=DISTNAME,
    version=VERSION,
    url=URL,
    project_urls=PROJECT_URLS,
    author=AUTHORS,
    author_email=AUTHORS_EMAIL,
    maintainer=AUTHORS,
    maintainer_email=AUTHORS_EMAIL,
    download_url=DOWNLOAD_URL,
    license=LICENSE,
    packages=find_packages(),  # ["dl85", "dl85.classifiers", "dl85.errors"],
    keywords=KEYWORDS,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    classifiers=CLASSIFIERS,
    setup_requires=SETUP_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    # extras_require=EXTRAS_REQUIRE,
    zip_safe=True,  # the package can run out of an .egg file
    ext_modules=cythonize(
        [dl85_extension],
        compiler_directives={"language_level": "3"}
    )
)
