import ctypes as C  # NOQA
import io
import pathlib
import typing

import obspy
import numpy as np

from . import utils

_f_types = (typing.Union[io.RawIOBase, str, pathlib.Path],)


def _buffer_proxy(
    filename_or_buf: _f_types,
    function: typing.Callable,
    reset_fp: bool = True,
    *args: typing.Any,
    **kwargs: typing.Any,
):
    """
    Calls a function with an open file or file-like object as the first
    argument. If the file originally was a filename, the file will be
    opened, otherwise it will just be passed to the underlying function.

    :param filename_or_buf: File to pass.
    :type filename_or_buf: str, open file, or file-like object.
    :param function: The function to call.
    :param reset_fp: If True, the file pointer will be set to the initial
        position after the function has been called.
    :type reset_fp: bool
    """
    # String or pathlib.Path => Open file
    if isinstance(filename_or_buf, (str, pathlib.Path)):
        with open(filename_or_buf, "rb") as fh:
            return function(fh, *args, **kwargs)

    # Otherwise it must have a tell method.
    position = filename_or_buf.tell()

    # Catch the exception to worst case be able to reset the file pointer.
    try:
        ret_val = function(filename_or_buf, *args, **kwargs)
    except Exception:
        filename_or_buf.seek(position, 0)
        raise
    # Always reset if reset_fp is True.
    if reset_fp:
        filename_or_buf.seek(position, 0)
    return ret_val


def _is_mseed3(filename: _f_types) -> bool:
    return _buffer_proxy(
        filename_or_buf=filename, function=_buffer_is_mseed3, reset_fp=True
    )


def _buffer_is_mseed3(handle: io.RawIOBase) -> bool:
    return handle.read(2) == b"MS"


def _read_mseed3(
    filename: _f_types,
    headonly: bool = False,
    starttime: typing.Optional[obspy.UTCDateTime] = None,
    endtime: typing.Optional[obspy.UTCDateTime] = None,
    verbose: bool = False,
    **kwargs,
) -> obspy.Stream():
    # Don't even bother passing on the extra kwargs - this should really be
    # cleaned up on ObsPy's side.
    return _buffer_proxy(
        filename_or_buf=filename,
        function=_buffer_read_mseed3,
        reset_fp=False,
        headonly=headonly,
        starttime=starttime,
        endtime=endtime,
        verbose=verbose,
    )


def _buffer_read_mseed3(
    handle: io.RawIOBase,
    headonly: bool = False,
    starttime: typing.Optional[obspy.UTCDateTime] = None,
    endtime: typing.Optional[obspy.UTCDateTime] = None,
    verbose: bool = False,
) -> obspy.Stream:
    r = utils._lib.mstl3_init(C.c_void_p())

    buffer = np.fromfile(handle, dtype=np.uint8)

    flags = utils._MSF_SKIPNOTDATA
    if not headonly:
        flags |= utils._MSF_UNPACKDATA

    utils._lib.mstl3_readbuffer(
        C.pointer(r),
        buffer,
        buffer.size,
        # Default time and sample rate tolerance.
        -1.0,
        -1.0,
        # splitversion - always split!
        1,
        # flags
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
            st.traces.append(
                _trace_segment_to_trace(
                    s,
                    source_identifier=t.sid.decode(),
                    publication_version=t.pubversion,
                )
            )
            current_segment = current_segment.contents.next
        current_trace = current_trace.contents.next
    return st


def _trace_segment_to_trace(
    t_s, source_identifier: str, publication_version: int
) -> obspy.Trace:
    tr = obspy.Trace()

    # Fill in headers.
    tr.stats.starttime = obspy.UTCDateTime(t_s.starttime / 1e9)
    tr.stats.sampling_rate = t_s.samprate

    tr.stats.mseed3 = obspy.core.AttribDict()
    tr.stats.mseed3.source_identifier = source_identifier
    tr.stats.mseed3.publication_version = publication_version
    tr.stats.mseed3.source_identifier = source_identifier

    tr.stats.network, tr.stats.station, tr.stats.location, tr.stats.channel = utils._source_id_to_nslc(
        sid=source_identifier
    )

    dtype = utils.SAMPLE_TYPES[t_s.sampletype]

    itemsize = dtype.itemsize
    if not isinstance(itemsize, int):
        itemsize = dtype().itemsize

    arr = np.ctypeslib.as_array(
        t_s.datasamples, shape=(t_s.numsamples * itemsize,)
    )
    arr.dtype = dtype

    # Make a copy as the default datasamples are freed. Maybe the freeing is
    # not necessary? Have to check.
    new_array = np.empty_like(arr)
    np.copyto(src=arr, dst=new_array)
    tr.data = new_array
    return tr
