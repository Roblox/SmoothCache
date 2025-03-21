# SmoothCache/setup.py

from setuptools import setup, find_packages

packages = find_packages()
print("Packages found:", packages)

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name='dit-smoothcache',
    version='v0.1.1',
    description='Training-free acceleration toolkit for Diffusion Transformer pipelines',
    packages=packages,
    author='Roblox Core AI',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Roblox/SmoothCache",    
    install_requires=[
        "torch>=2.0.0",
    ],    
)
