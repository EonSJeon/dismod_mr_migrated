from setuptools import setup, find_packages

setup(
    name="dismod_mr",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "pymc",
        "networkx",
        "pytensor",
    ],
) 