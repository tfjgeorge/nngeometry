from setuptools import setup

setup(name='nngeometry',
      version='0.1',
      description='Manipulate geometry matrices in Pytorch',
      url='https://github.com/tfjgeorge/nngeometry',
      author='tfjgeorge',
      author_email='tfjgeorge@gmail.com',
      license='MIT',
      packages=['nngeometry'],
      install_requires=['torch>=1.0.0'],
      zip_safe=False)
