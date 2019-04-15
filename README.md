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
$ git submodule init && git submodule update

# Make sure you have a C compiler.
$ pip install -v -e .

# Test it.
$ py.test
```

## ToDo

* Hook into `ms_log()` so that diagnostic messages, warnings and exceptions are
  raised from Python and thus can caught and handled. Similar to what ObsPy is
  currently doing for `libmseed` 2.
* No way to read/write any of the extra header fields yet.
* More flexible way to set the sample rate and timing tolerances.

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

The `publication_version` will also be used during writing so a separate
`publication_version` can be written for every `Trace.`

### Extra Arguments During Reading

* `headonly`: Determines whether or not to unpack the data or just read the headers.
* `starttime`: Only read data samples after or at the start time.
* `endtime`: Only read data samples before or at the end time.
* `sidpattern`: Only keep records whose SID pattern match the given pattern. Please note that the pattern must also includes the namespaces, e.g. `XFDSN` and any potential agency.
* `publication_versions`: A list of publication versions to retain. If not given, all publication versions will be read.
* `parse_record_level_metadata`: If True, per-record meta-data will be parsed
  and stored in each Trace's stats attribute dictionary. This is potentially
  much slower.
* `verbose`: Controls verbosity - passed to `libmseed`.

**Example:**

```python
st = obspy.read("file.ms3", starttime=obspy.UTCDateTime(2012, 1, 1),
                sidpattern="XFDSN:TA_*", publication_version=[1, 2, 3])
```

### Extra Arguments During Writing


* `max_record_length`: Maximum record length.
* `publication_version`: Publication version for all traces if given. Will overwrite any per-trace settings.
* `record_level_flags`: Record level flags for every record. Will
  overwrite any per-trace settings if given.
* `record_level_extra_data`: Record level extra data that will be stored as a
  compact JSON string with each record. Will overwrite any per-trace settings.
* `encoding`: Data encoding. Must be compatible with the underlying dtype. If not given it will be chosen automatically. Int32 data will default to STEIM2 encoding.
* `verbose`: Controls verbosity - passed to `libmseed`.

**Example:**

```python
from libmseed3_obspy_plugin import utils

st.write("out.ms3", format="mseed3", encoding=utils.Encoding.STEIM1,
         publication_version=2)
```
