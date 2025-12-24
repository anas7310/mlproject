##Setup.py is used to build application as a package itself
from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'
def get_requirements(filename:str)->List[str]:
    """
        This function will return the list of requirements
    """
    requirements=[]

    with open(filename) as f:
        requirements=f.readlines()
        requirements =[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements

setup(
    name='mlproject',
    version='0.0.1',
    author='AnasKhan',
    author_email='yarkhananas42@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)


