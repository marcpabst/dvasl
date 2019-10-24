from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
     name='dvasl',  
     version='0.1',
     author="Marc Pabst",
     author_email="mapabst@cbs.mpg.de",
     packages=find_packages(),
     description="Searchlight-like analysis for fMRI data.",
     long_description=long_description
 )