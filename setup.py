from setuptools import setup

setup(name='nngeometry',
      version='0.3',
      description='Manipulate geometry matrices in Pytorch',
      url='https://github.com/tfjgeorge/nngeometry',
      author='tfjgeorge',
      author_email='tfjgeorge@gmail.com',
      license='MIT',
      packages=['nngeometry',
                'nngeometry.generator',
                'nngeometry.generator.jacobian',
                'nngeometry.object'],
      install_requires=['torch>=2.0.0'],
      zip_safe=False)
