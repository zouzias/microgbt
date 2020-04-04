import os
try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages


__version__ = "0.0.1"


def get_requirements(env):
    """Get requirements from requirements.txt file"""
    script_dir = os.path.dirname(__file__)
    with open(os.path.join(script_dir, u"requirements-{}.txt".format(env))) as fp:
        return [x.strip() for x in fp.read().split("\n") if not x.startswith("#")]


setup_requirements = ['pytest-runner']

setup(
    name="microgbtpy",
    version=__version__,
    author="Anastasios Zouzias",
    url="https://github.com/zouzias/microgbt",
    description="microgbt is a minimalistic Gradient Boosting Trees implementation",
    license="Apache 2.0 license",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    include_package_data=True,
    keywords="gradient boosting trees",
    setup_requires=setup_requirements,
    test_suite='tests',
    zip_safe=False,
    packages=find_packages(),
    package_dir={"": "."},
    package_data={"": ["microgbtpy*.so"]},
)
