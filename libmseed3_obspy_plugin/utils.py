import ctypes as C  # NOQA
import pathlib
import re
import typing
import warnings

import numpy as np

_lib_folder = pathlib.Path(__file__).parent / "lib"
_lib_files = list(_lib_folder.glob("libmseed.*.so"))
if not _lib_files:
    raise ValueError("Could not find a compiled shared libmseed library.")
elif len(_lib_files) > 1:
    warnings.warn(
        f"Found multiple compiled libmseed libraries in {_lib_folder}. Will "
        f"use the first one."
    )
_lib_file = _lib_files[0]
_lib = C.CDLL(_lib_file)


# Some #defines. No way to get them from the shared library.
_LM_SIDLEN = 64
_MSF_UNPACKDATA = 0x0001
_MSF_SKIPNOTDATA = 0x0002
_MSF_VALIDATECRC = 0x0004
_MSF_SEQUENCE = 0x0008
_MSF_FLUSHDATA = 0x0010
_MSF_ATENDOFFILE = 0x0020

# Also some typedefs
_nstime_t = C.c_longlong


SAMPLE_TYPES = {
    b"a": np.dtype("|S1"),
    b"i": np.int32,
    b"f": np.float32,
    b"d": np.float64,
}


class MS3TraceSeg(C.Structure):
    pass


MS3TraceSeg._fields_ = [
    ("starttime", _nstime_t),
    ("endtime", _nstime_t),
    ("samprate", C.c_double),
    ("samplecnt", C.c_longlong),
    ("datasamples", C.POINTER(C.c_uint8)),
    ("numsamples", C.c_longlong),
    ("sampletype", C.c_char),
    ("prvtptr", C.c_void_p),
    ("prev", C.POINTER(MS3TraceSeg)),
    ("next", C.POINTER(MS3TraceSeg)),
]


class MS3TraceID(C.Structure):
    pass


MS3TraceID._fields_ = [
    ("sid", C.c_char * _LM_SIDLEN),
    ("pubversion", C.c_uint8),
    ("earliest", _nstime_t),
    ("latest", _nstime_t),
    ("prvtptr", C.c_void_p),
    ("numsegments", C.c_uint),
    ("first", C.POINTER(MS3TraceSeg)),
    ("last", C.POINTER(MS3TraceSeg)),
    ("next", C.POINTER(MS3TraceID)),
]


class MS3TraceList(C.Structure):
    _fields_ = [
        ("numtraces", C.c_uint),  # Number of traces in list.
        ("traces", C.POINTER(MS3TraceID)),  # Pointer to list of traces.
        (
            "last",
            C.POINTER(MS3TraceID),
        ),  # Pointer to last modified trace in list.
    ]


_lib.mstl3_init.argtypes = [C.c_void_p]
_lib.mstl3_init.restype = C.POINTER(MS3TraceList)

_lib.mstl3_free.argtypes = [C.POINTER(C.POINTER(MS3TraceList)), C.c_int8]
_lib.mstl3_free.restype = C.c_void_p

_lib.mstl3_readbuffer.argtypes = [
    # Destination trace list.
    C.POINTER(C.POINTER(MS3TraceList)),
    # Data source.
    np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags="C_CONTIGUOUS"),
    # buffer length
    C.c_ulonglong,
    # time tolerance
    C.c_double,
    # Sample rate tolerance
    C.c_double,
    # Split version
    C.c_int8,
    # flags,
    C.c_uint,
    # verbose,
    C.c_int8,
]
_lib.mstl3_readbuffer.restype = C.c_longlong

_SID_REGEX = re.compile(
    r"""
    ([A-Z]+:)*           # Any number of agencies/namespaces
    (?P<net>[A-Z]*)       # Network code
    _
    (?P<sta>[A-Z]*)       # Station code
    _
    (?P<loc>[A-Z]*)       # Location code
    _
    (?P<band>[A-Z]*)      # Channel badn
    _
    (?P<source>[A-Z]*)    # Channel source
    _
    (?P<position>[A-Z]*)  # Channel position
    """,
    re.X,
)


def _source_id_to_nslc(sid: str) -> typing.Tuple[str, str, str, str]:
    """
    Parses a source identifier to network, station, location, channel.

    This is a port of ms_sid2nslc() in libmseed but writing it with a regex
    is simpler than wrapping the code.
    """
    m = re.match(_SID_REGEX, sid)
    if not m:
        raise ValueError(
            f"Source identifiers '{sid}' did not match the "
            "expected pattern."
        )

    return (
        m.group("net"),
        m.group("sta"),
        m.group("loc"),
        m.group("band") + m.group("source") + m.group("position"),
    )
