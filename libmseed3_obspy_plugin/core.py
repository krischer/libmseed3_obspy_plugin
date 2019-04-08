import ctypes as C  # NOQA
import pathlib
import typing

import obspy
import numpy as np

from . import utils


def read(handle: typing.Union[str, pathlib.Path]) -> obspy.Stream:
    if not hasattr(handle, "flush"):
        with open(handle, "rb") as fh:
            return _read_buffer(np.fromfile(fh, dtype=np.uint8))
    return _read_buffer(np.fromfile(handle, dtype=np.uint8))


def _read_buffer(
    buffer: bytes, unpack_data: bool = True, verbose: bool = False
) -> obspy.Stream:
    r = utils._lib.mstl3_init(C.c_void_p())

    flags = utils._MSF_SKIPNOTDATA
    if unpack_data:
        flags |= utils._MSF_UNPACKDATA

    utils._lib.mstl3_readbuffer(
        C.pointer(r),
        buffer,
        buffer.size,
        -1.0,
        -1.0,
        1,
        flags,
        # verbose
        1 if bool else 0,
    )
    st = _tracelist_to_stream(r)
    utils._lib.mstl3_free(C.pointer(r), 0)
    return st


def _tracelist_to_stream(t_l):
    st = obspy.Stream()

    current_trace = t_l.contents.traces
    for _ in range(t_l.contents.numtraces):
        t = current_trace.contents
        current_segment = t.first
        for _ in range(t.numsegments):
            s = current_segment.contents
            st.traces.append(_trace_segment_to_trace(s, id=t.sid.decode()))
            current_segment = current_segment.contents.next
        current_trace = current_trace.contents.next
    return st


def _trace_segment_to_trace(t_s, id: str) -> obspy.Trace:
    tr = obspy.Trace()
    tr.stats.starttime = obspy.UTCDateTime(t_s.starttime / 1e9)
    tr.stats.sampling_rate = t_s.samprate

    dtype = utils.SAMPLE_TYPES[t_s.sampletype]
    itemsize = dtype().itemsize

    arr = np.ctypeslib.as_array(
        t_s.datasamples, shape=(t_s.numsamples * itemsize,)
    )
    arr.dtype = dtype

    new_array = np.empty_like(arr)

    np.copyto(src=arr, dst=new_array)
    tr.data = new_array
    return tr
