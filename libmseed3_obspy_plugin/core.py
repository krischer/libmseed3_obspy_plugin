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
        1 if verbose else 0,
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

    nslc = utils._source_id_to_nslc(sid=source_identifier)
    tr.stats.network = nslc[0]
    tr.stats.station = nslc[1]
    tr.stats.location = nslc[2]
    tr.stats.channel = nslc[3]

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


def _write_mseed3(
    stream: obspy.Stream(),
    filename: _f_types,
    max_record_length: int = 4096,
    publication_version: typing.Optional[int] = None,
    encoding: typing.Optional[typing.Union[utils.Encoding, str]] = None,
    verbose: bool = False,
) -> None:
    # Map encoding string to enumerated value.
    if isinstance(encoding, str):
        encoding = utils.Encoding[encoding.upper()]

    # Don't even bother passing on the extra kwargs - this should really be
    # cleaned up on ObsPy's side.
    return _buffer_proxy(
        filename_or_buf=filename,
        function=_buffer_write_mseed3,
        stream=stream,
        max_record_length=max_record_length,
        publication_version=publication_version,
        verbose=verbose,
        encoding=encoding,
    )


def _buffer_write_mseed3(
    handle: io.RawIOBase,
    stream: obspy.Stream,
    max_record_length: int,
    encoding: typing.Optional[utils.Encoding] = None,
    publication_version: typing.Optional[int] = None,
    verbose: bool = False,
) -> None:
    # Check all encodings - this is redundant but will raise an error if
    # encoding or dtypes are invalid/incompatible.
    [
        utils._get_or_check_encoding(data=tr.data, encoding=encoding)
        for tr in stream
    ]

    # We'll do this separately for every single trace - this has the advantage
    # that we can set encoding and what not separately for every trace.
    for trace in stream:
        _buffer_write_mseed3_trace(
            handle=handle,
            trace=trace,
            max_record_length=max_record_length,
            encoding=encoding,
            publication_version=publication_version,
            verbose=verbose,
        )


def _buffer_write_mseed3_trace(
    handle: io.RawIOBase,
    trace: obspy.Trace,
    max_record_length: int,
    encoding: typing.Optional[utils.Encoding] = None,
    publication_version: typing.Optional[int] = None,
    verbose: bool = False,
) -> None:

    encoding = utils._get_or_check_encoding(data=trace.data, encoding=encoding)

    # The only case in which we'll convert data types if for int16.
    if trace.data.dtype == np.int16:
        trace = trace.copy()
        trace.data = np.require(trace.data, dtype=np.int32)

    trace_id = _trace_to_trace_id(
        trace=trace, publication_version=publication_version
    )

    trace_list = utils.MS3TraceList(numtraces=1)
    trace_list.traces.contents = trace_id
    trace_list.last.contents = trace_id

    packed_samples = C.c_longlong(0)

    if encoding is None:
        encoding = 1

    # Callback function for mstl3_pack to actually write the file
    def record_handler(record, reclen, _stream):
        handle.write(record[0:reclen])

    # Define Python callback function for use in C function
    rec_handler = C.CFUNCTYPE(None, C.POINTER(C.c_char), C.c_int, C.c_void_p)(
        record_handler
    )

    utils._lib.mstl3_pack(
        C.pointer(trace_list),
        rec_handler,
        # Pointer passed to the callback function.
        C.c_void_p(),
        # Maximum record length.
        max_record_length,
        # Encoding.
        encoding.value,
        # The number of packed samples - returned to the caller.
        C.pointer(packed_samples),
        # flags. Always flush the data - seems to be what we want in ObsPy.
        utils._MSF_FLUSHDATA,
        # verbose,
        1 if verbose else 0,
        # Pointer to a packed JSON string.
        C.c_char_p(),
    )


def _trace_to_trace_id(
    trace: obspy.Trace, publication_version: typing.Optional[int] = None
):
    # Deal with the publication version. Code is a bit ugly but not much to be
    # done about it.
    pub_ver = None
    # Get it from the stats.mseed3 dict.
    if "mseed3" in trace.stats and "publication_version" in trace.stats.mseed3:
        pub_ver = trace.stats.mseed3.publication_version
    # If given, always use that.
    if publication_version:
        pub_ver = publication_version
    # Default to 1.
    if pub_ver is None:
        pub_ver = 1

    bytecount = trace.data.itemsize * trace.data.size
    datasamples = utils._libmseed_memory.contents.malloc(bytecount)
    datasamples = C.cast(datasamples, C.POINTER(C.c_uint8))
    # Copy the data as libmseed does free the memory.
    C.memmove(datasamples, trace.data.ctypes.get_data(), bytecount)

    trace_seg = utils.MS3TraceSeg(
        starttime=trace.stats.starttime.ns,
        endtime=trace.stats.endtime.ns,
        samprate=trace.stats.sampling_rate,
        samplecnt=trace.stats.npts,
        datasamples=datasamples,
        numsamples=trace.stats.npts,
        sampletype=utils.INV_SAMPLE_TYPES[trace.data.dtype],
    )

    trace_id = utils.MS3TraceID(
        sid=utils._nslc_to_source_id(
            network=trace.stats.network,
            station=trace.stats.station,
            location=trace.stats.location,
            channel=trace.stats.channel,
        ).encode(),
        earliest=trace.stats.starttime.ns,
        latest=trace.stats.endtime.ns,
        pubversion=pub_ver,
        numsegments=1,
    )
    trace_id.first.contents = trace_seg
    trace_id.last.contents = trace_seg

    return trace_id
