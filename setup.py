from setuptools import setup, find_packages
"""
Install project directory to environment from the same level of this file with:
pip install -e .

Note: -e will track the changes to your packages, check with pip list 
"""
setup(name='project', version='1.0', packages=find_packages())