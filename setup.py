from setuptools import setup, find_packages

setup(
    name="dismod_mr",
    version="0.1",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.2",
        "pandas>=2.0.0",
        "pymc>=5.3.0",
        "networkx>=3.2.1",
        "pytensor>=2.11.1",
    ],
) 