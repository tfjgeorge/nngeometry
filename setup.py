from setuptools import setup, find_packages

setup(name='nngeometry',
      version='0.3.2',
      description='{KFAC,EKFAC,Diagonal,Implicit} Fisher Matrices and finite width NTKs in PyTorch',
      url='https://github.com/tfjgeorge/nngeometry',
      author='tfjgeorge',
      author_email='tfjgeorge@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=['torch>=2.0.0','torchvision>=0.9.1'],
      zip_safe=False)