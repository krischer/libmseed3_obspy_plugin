# libmseed3 ObsPy Plug-in

ObsPy Plug-in to use `libmseed` >= 3.0.0 to read and write data in
miniSEED3/XSEED/whatever it is called (will be called "mseed3" inside the
plug-in cause I had to choose a name) format.

Will likely migrate to core ObsPy once the format, definition, and library
has stabilized.

**STATUS:** Alpha

---

**DON'T USE IN PRODUCTION - NEITHER THE FORMAT NOR THE LIBRARY ARE FINALIZED!!!**

---


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


### Record Level Flags

Record level flags are dealt with using the `utils.RecordFlag` object:

```python
from libmseed3_obspy_plugins.utils import RecordFlag

f = RecordFlag.calibration_signal_present | RecordFlag.clock_locked

# You can set them as part of a Trace's stats attribute.
st[0].stats.mseed3.record_level_flags = f

# Or globally - they will overwrite the per-trace settings.
st.write("out.ms3", format="mseed3", record_level_flags=f)
```

By default they will not be read, passing the `parse_record_level_metadata` to the `read()` function will add them to the trace metadata. It will add one entry per record, thus potentially a lot.

```python
>>> tr = obspy.read("file.ms3", parse_record_level_metadata=True)[0]
>>> tr.stats.mseed3.record_level_metadata
[AttribDict({
    'starttime': UTCDateTime(2009, 8, 24, 0, 20, 3),
    'endtime': UTCDateTime(2009, 8, 24, 0, 20, 8, 30000),
    'flags': <RecordFlag.clock_locked|calibration_signal_present: 5>}),
 AttribDict({
     'starttime': UTCDateTime(2009, 8, 24, 0, 20, 8, 40000),
     'endtime': UTCDateTime(2009, 8, 24, 0, 20, 13, 70000),
     'flags': <RecordFlag.clock_locked|calibration_signal_present: 5>})]
```

### Record Level JSON Metadata

An arbitrary JSON document can be stored per record. This plug-in tranparently maps dictionaries to the JSON data.

```python
# You can set them as part of a Trace's stats attribute.
st[0].stats.mseed3.record_level_extra_data = {"some_extra_data": 1.2}

# Or globally - they will overwrite the per-trace settings.
st.write("out.ms3", format="mseed3",
         record_level_extra_data={"a": True, "b": [1, 2, 3]})
```

Parsing them again requires the `parse_record_level_metadata` keyword argument to be set. And, as with the flags, this is potentially expensive and might result in a lot of meta data.

```python
>>> tr = obspy.read("file.ms3", parse_record_level_metadata=True)[0]
>>> tr.stats.mseed3.record_level_metadata
[AttribDict({
    'starttime': UTCDateTime(2009, 8, 24, 0, 20, 3),
    'endtime': UTCDateTime(2009, 8, 24, 0, 20, 8),
    'flags': <RecordFlag.0: 0>,
    'extra_data': AttribDict({'a': True, 'b': [1, 2, 3]})}),
 AttribDict({
     'starttime': UTCDateTime(2009, 8, 24, 0, 20, 8, 10000),
     'endtime': UTCDateTime(2009, 8, 24, 0, 20, 13, 10000),
     'flags': <RecordFlag.0: 0>,
     'extra_data': AttribDict({'a': True, 'b': [1, 2, 3]})}),
```