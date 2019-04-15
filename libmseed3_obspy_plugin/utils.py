import ctypes as C  # NOQA
import enum
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


class Encoding(enum.Enum):
    ASCII = 0
    INT16 = 1
    INT32 = 3
    FLOAT32 = 4
    FLOAT64 = 5
    STEIM1 = 10
    STEIM2 = 11


# Maps dtype to allowed encodings. The first one is always the default encoding
# for that data type.
DTYPE_TO_ENCODING = {
    np.dtype("int16"): [Encoding.INT16],
    np.dtype("int32"): [Encoding.STEIM2, Encoding.STEIM1, Encoding.INT32],
    np.dtype("float32"): [Encoding.FLOAT32],
    np.dtype("float64"): [Encoding.FLOAT64],
    np.dtype("|S1"): [Encoding.ASCII],
}


def _get_or_check_encoding(
    data: np.ndarray, encoding: typing.Optional[Encoding] = None
) -> Encoding:
    """
    Helper function returning the encoding for a given data array.

    If encoding is given it will check if the encoding is compatible with the
    data, otherwise it will raise an Exception.
    """
    if data.dtype not in DTYPE_TO_ENCODING:
        msg = (
            f"dtype {data.dtype} not allowed. Please convert to one of the "
            "supported dtypes: "
            f"{', '.join(str(i) for i in DTYPE_TO_ENCODING.keys())}"
        )
        raise TypeError(msg)
    allowed_encodings = DTYPE_TO_ENCODING[data.dtype]
    if encoding is not None:
        if encoding not in allowed_encodings:
            msg = (
                f"Encoding {encoding} is not compatible with dtype "
                f"{data.dtype}. Please choose a different encoding or "
                "convert your data."
            )
            raise ValueError(msg)
        return encoding
    return allowed_encodings[0]


SAMPLE_TYPES = {
    b"a": np.dtype("|S1"),
    b"i": np.dtype("int32"),
    b"f": np.dtype("float32"),
    b"d": np.dtype("float64"),
}

INV_SAMPLE_TYPES = {value: key for key, value in SAMPLE_TYPES.items()}


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


class MS3Record(C.Structure):
    _fields_ = [
        ("record", C.c_char_p),
        ("reclen", C.c_int),
        ("swapflag", C.c_uint8),
        ("sid", C.c_char * _LM_SIDLEN),
        ("formatversion", C.c_uint8),
        ("flags", C.c_uint8),
        ("starttime", _nstime_t),
        ("samprate", C.c_double),
        ("encoding", C.c_int8),
        ("pubversion", C.c_uint8),
        ("samplecnt", C.c_longlong),
        ("crc", C.c_uint),
        ("extralength", C.c_ushort),
        ("datalength", C.c_ushort),
        ("extra", C.c_char_p),
        ("datasamples", C.POINTER(C.c_uint8)),
        ("numsamples", C.c_longlong),
        ("sampletype", C.c_char),
    ]


class MS3Tolerance(C.Structure):
    _fields_ = [
        ("time", C.CFUNCTYPE(C.c_double, C.POINTER(MS3Record))),
        ("samprate", C.CFUNCTYPE(C.c_double, C.POINTER(MS3Record))),
    ]


_lib.mstl3_init.argtypes = [C.c_void_p]
_lib.mstl3_init.restype = C.POINTER(MS3TraceList)

_lib.mstl3_free.argtypes = [C.POINTER(C.POINTER(MS3TraceList)), C.c_int8]
_lib.mstl3_free.restype = C.c_void_p

_lib.msr3_pack.argtypes = [
    # Source record.
    C.POINTER(MS3Record),
    # Callback function to do the actual writing.
    C.CFUNCTYPE(None, C.POINTER(C.c_char), C.c_int, C.c_void_p),
    # Pointer passed to the callback function.
    C.c_void_p,
    # The number of packed samples - returned to the caller.
    C.POINTER(C.c_longlong),
    # flags,
    C.c_uint,
    # verbosity
    C.c_int8,
]
_lib.msr3_pack.restype = C.c_int


class MS3SelectTime(C.Structure):
    pass


MS3SelectTime._fields_ = [
    ("starttime", _nstime_t),
    ("endtime", _nstime_t),
    ("next", C.POINTER(MS3SelectTime)),
]


class MS3Selections(C.Structure):
    pass


MS3Selections._fields_ = [
    ("sidpattern", C.c_char * 100),
    ("timewindows", C.POINTER(MS3SelectTime)),
    ("next", C.POINTER(MS3Selections)),
    ("pubversion", C.c_uint8),
]


_lib.mstl3_readbuffer_selection.argtypes = [
    # Destination trace list.
    C.POINTER(C.POINTER(MS3TraceList)),
    # Data source.
    np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags="C_CONTIGUOUS"),
    # buffer length
    C.c_ulonglong,
    # Split version
    C.c_int8,
    # flags,
    C.c_uint,
    # Tolerance callbacks.
    C.POINTER(MS3Tolerance),
    # Selections.
    C.POINTER(MS3Selections),
    # verbose,
    C.c_int8,
]
_lib.mstl3_readbuffer_selection.restype = C.c_longlong

_lib.mstl3_pack.argtypes = [
    # Source trace list.
    C.POINTER(MS3TraceList),
    # Callback function to do the actual writing.
    C.CFUNCTYPE(None, C.POINTER(C.c_char), C.c_int, C.c_void_p),
    # Pointer passed to the callback function.
    C.c_void_p,
    # Maximum record length.
    C.c_int,
    # Encoding.
    C.c_int8,
    # The number of packed samples - returned to the caller.
    C.POINTER(C.c_longlong),
    # flags.
    C.c_uint,
    # verbose,
    C.c_int8,
    # Pointer to a packed JSON string.
    C.c_char_p,
]
# Returns the number of created records.
_lib.mstl3_pack.restype = C.c_int


class LibmseedMemory(C.Structure):
    _fields_ = [
        ("malloc", C.CFUNCTYPE(C.c_void_p, C.c_size_t)),
        ("realloc", C.CFUNCTYPE(C.c_void_p, C.c_void_p, C.c_size_t)),
        ("free", C.CFUNCTYPE(None, C.c_void_p)),
    ]


_libmseed_memory = C.cast(_lib.libmseed_memory, C.POINTER(LibmseedMemory))


_SID_REGEX = re.compile(
    r"""
    ([A-Z]+:)*           # Any number of agencies/namespaces
    (?P<net>[A-Z]*)       # Network code
    _
    (?P<sta>[A-Z]*)       # Station code
    _
    (?P<loc>[A-Z]*)       # Location code
    _
    # Now either allow for separate band, source, position codes or a single
    # all encompassing one.
    (
        (
            (?P<band>[A-Z]*)      # Channel band
            _
            (?P<source>[A-Z]*)    # Channel source
            _
            (?P<position>[A-Z]*)  # Channel position
        )
        |
        (
            (?P<bsp>[A-Z]*)  # BSP
        )
    )

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

    b, s, p = m.group("band"), m.group("source"), m.group("position")

    if any((b, s, p)):
        bsp = b + s + p
    else:
        bsp = m.group("bsp")

    return (
        m.group("net") or "",
        m.group("sta") or "",
        m.group("loc") or "",
        bsp or "",
    )


def _nslc_to_source_id(
    network: str, station: str, location: str, channel: str
) -> str:
    """
    Parses network, station, location, channel codes to a source identifier.

    This is a port of ms_nslc2sid() in libmseed but writing it in Python is
    simpler than wrapping it.
    """
    if len(channel) == 3:
        bsl = f"{channel[0]}_{channel[1]}_{channel[2]}"
    else:
        bsl = channel
    return f"XFDSN:{network}_{station}_{location}_{bsl}"


def _verbosity_to_int(verbose: typing.Union[bool, int]) -> int:
    if verbose is False:
        return 0
    elif verbose is True:
        return 1
    else:
        return verbose
