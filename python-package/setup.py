from setuptools import setup

__version__ = "0.0.1"

setup(
    name="microgbtpy",
    author="Anastasios Zouzias",
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
    url="https://github.com/zouzias/microgbt",
    zip_safe=False,
    version=__version__,  # specified elsewhere
    packages=[""],
    package_dir={"": "."},
    package_data={"": ["microgbtpy.cpython-*-darwin.so"]},
)
