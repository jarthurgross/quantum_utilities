from setuptools import setup

requires = [
        'numpy >= 1.13',
        'seaborn',
         ]

setup(name='quantum_utilities',
      install_requires=requires,
      packages=['quantum_utilities'],
      package_dir={'': 'src'},
     )
