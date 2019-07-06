from setuptools import setup, Extension

__version__ = "0.0.1"

setup(
    name='microgbtpy',
    version=__version__,  # specified elsewhere
    packages=[''],
    package_dir={'': '.'},
    package_data={'': ['microgbtpy.cpython-35m-darwin.so']},
)
