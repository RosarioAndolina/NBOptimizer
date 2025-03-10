from setuptools import setup, find_packages

setup(
    name="nboptimizer",
    version="0.1.0",
    description="A Python package for numerical optimization using Numba.",
    author="Rosario Andolina",
    author_email="andolinarosario@gmail.com",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "numba>=0.55.0",
    ],
    python_requires=">=3.8",
)

