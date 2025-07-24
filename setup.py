'''setup.py ek Python file hoti hai jo kisi Python package ko installable banati hai. Jab aap koi module ya library banate ho (jaise ML project ya API), 
to agar usse kisi aur system par install karna ho pip install ke through, to setup.py zaroori hota hai.'''

from setuptools import setup, find_packages
from typing import List

# HYPEN_E_DOT = '-e .'

def get_requirements(file_path:str) -> List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        #if HYPEN_E_DOT in requirements:
            #requirements.remove(HYPEN_E_DOT)

        return requirements


setup(
    name="California_HousePrice_prediction",
    version="0.0.1",
    author="Sujen",
    author_email="sujenpurty4@gmail.com",
    install_requires=get_requirements('requirements.txt'),
    packages=find_packages()
)