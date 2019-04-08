import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="libmseed3-obspy-plugin",
    version="0.0.1",
    author="Lion Krischer",
    author_email="lion.krischer@gmail.com",
    description="ObsPy plug-in to read MiniSEED3 data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/krischer/libmseed3_obspy_plugin",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
