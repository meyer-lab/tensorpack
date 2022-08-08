try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

setup(name='tensorpack',
      version='0.1.0',
      description='A collection of tensor methods from the Meyer lab.',
      url='https://github.com/meyer-lab/tensorpack',
      license='MIT',
      packages=find_packages(exclude=['doc']),
      install_requires=['numpy', 'tensorly', 'scikit-learn'])
