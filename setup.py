from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
import platform

source_files = ["wrapping/dl85Optimizer.pyx",
                "wrapping/src/codes/data.cpp",
                "wrapping/src/codes/dataContinuous.cpp",
                "wrapping/src/codes/dataBinary.cpp",
                "wrapping/src/codes/dl85.cpp",
                "wrapping/src/codes/experror.cpp",
                "wrapping/src/codes/globals.cpp",
                "wrapping/src/codes/lcm_pruned.cpp",
                "wrapping/src/codes/lcm_iterative.cpp",
                "wrapping/src/codes/query.cpp",
                "wrapping/src/codes/query_best.cpp",
                "wrapping/src/codes/query_totalfreq.cpp",
                "wrapping/src/codes/trie.cpp",
                "wrapping/src/codes/dataBinaryPython.cpp"]

args = ['-std=c++14']
if platform.system() == 'Darwin':
    args.append('-mmacosx-version-min=10.12')

dl85_extension = Extension(
    name="dl85Optimizer",
    language="c++",
    sources=source_files,
    # libraries=["dl85"],  # the name of .a library if external library is used
    # library_dirs=["cmake-build-release"],  # the relative path of the library if necessary
    include_dirs=["wrapping/src/headers"],  # path for headers
    extra_compile_args=args,  # ['-std=c++14', '-mmacosx-version-min=10.12'],#'-v' option for verbose during compilation
    extra_link_args=args  # '-v' option for verbose during linkage
)

setup(
    name="dl85",
    version="0.0.1",
    url="https://github.com/aglingael/dl85_dist_source",
    project_urls={
        "Source on github": "https://github.com/aglingael/dl85_dist_source",
        "Documentation": "https://github.com/aglingael/dl85_dist_source/tree/master/docs/build/html/index.html",
    },
    author="Gael Aglin",
    author_email='aglingael@gmail.com',
    license="LICENSE.txt",
    packages=["dl85", "dl85.classifiers", "dl85.errors"],
    keywords="optimal decision trees",
    description='A package to build optimal decision trees classifier',
    long_description=open('README.md').read(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    setup_requires=['cython'],
    install_requires=[
        "numpy",
        "cython>=0.25",
        "setuptools>=18.0",
        "pytest",
        "scikit-learn",
    ],
    ext_modules=cythonize(
        [dl85_extension],
        compiler_directives={"language_level": "3"}
    )
)
