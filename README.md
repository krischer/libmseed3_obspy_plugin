# libmseed3 ObsPy Plug-in

ObsPy Plug-in to use `libmseed` >= 3.0.0 to read and write data in
miniSEED3/XSEED/whatever it is called (will be called "mseed3" inside the
plug-in cause I had to choose a name) format.

Will likely migrate to core ObsPy once the format, definition, and library
has stabilized.

**STATUS:** Beta

## Installation

Install ObsPy, this plug-in only works with Python 3.6 and 3.7.

```bash
$ git clone https://github.com/krischer/libmseed3_obspy_plugin.git
$ cd libmseed3_obspy_plugin
$ git submodule init && git submodule updatgit submodule init && git submodule updatee

# Make sure you have a C compiler.
$ pip install -v -e .

# Test it.
$ py.test
```

## Usage

It will hook into ObsPy's normal I/O handling and you just use it as you would
use any format:

```python
>>> import obspy
# Read
>>> st = obspy.read("/path/to/file.ms3")
# Write
>>> st.write("out.ms3", format="mseed3")
```

It currently has two format specific attributes:

```python
>>> tr.stats.mseed3
AttribDict({'source_identifier': 'XFDSN:XX_TEST__L_H_Z',
            'publication_version': 1})
```
