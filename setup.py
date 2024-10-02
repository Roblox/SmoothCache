# SmoothCache/setup.py

from setuptools import setup, find_packages

packages = find_packages()
print("Packages found:", packages)

setup(
    name='SmoothCache',
    version='0.1',
    packages=packages,
    # Other setup parameters...
)
