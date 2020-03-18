from setuptools import setup

setup(name='nngeometry',
      version='0.1',
      description='Manipulate geometry matrices in Pytorch',
      url='#',
      author='tfjgeorge',
      author_email='tfjgeorge@gmail.com',
      license='MIT',
      packages=['nngeometry',
                'nngeometry.pspace',
                'nngeometry.fspace',
                'nngeometry.jacobian'],
      zip_safe=False)
