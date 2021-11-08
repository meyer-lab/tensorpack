from setuptools import setup, find_packages

setup(name='tensorPack',
      version='0.1',
      description='A collection of tensor methods from the Meyer lab.',
      url='https://github.com/meyer-lab/tensorPack',
      license='MIT',
      packages=find_packages(exclude=['doc']),
      install_requires=['numpy', 'tensorly', 'scikit-learn'])