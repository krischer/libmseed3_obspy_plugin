import copy
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
    file_mode: str,
    reset_fp: bool,
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
    :param file_mode: The mode in which to open the file if it has to be
        opened.
    :param reset_fp: If True, the file pointer will be set to the initial
        position after the function has been called.
    :type reset_fp: bool
    """
    # String or pathlib.Path => Open file
    if isinstance(filename_or_buf, (str, pathlib.Path)):
        with open(filename_or_buf, file_mode) as fh:
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
        filename_or_buf=filename,
        function=_buffer_is_mseed3,
        file_mode="rb",
        reset_fp=True,
    )


def _buffer_is_mseed3(handle: io.RawIOBase) -> bool:
    return handle.read(2) == b"MS"


def _read_mseed3(
    filename: _f_types,
    headonly: bool = False,
    starttime: typing.Optional[obspy.UTCDateTime] = None,
    endtime: typing.Optional[obspy.UTCDateTime] = None,
    sidpattern: typing.Optional[str] = None,
    publication_versions: typing.Optional[
        typing.Union[int, typing.List[int]]
    ] = None,
    verbose: typing.Union[bool, int] = False,
    **kwargs,
) -> obspy.Stream():
    """
    :param headonly: Determines whether or not to unpack the data or just
        read the headers.
    :param starttime: Only read data samples after or at the start time.
    :param endtime: Only read data samples before or at the end time.
    :param sidpattern: Only keep records whose SID pattern match the given
        pattern. Please note that the pattern must also includes the
        namespaces, e.g. ``XFDSN`` and any potential agency.
    :param publication_versions: A list of publication versions to retain. If
        not given, all publication versions will be read.
    :param verbose: Controls verbosity - passed to libmseed.
    """
    # Don't even bother passing on the extra kwargs - this should really be
    # cleaned up on ObsPy's side.
    return _buffer_proxy(
        filename_or_buf=filename,
        function=_buffer_read_mseed3,
        file_mode="rb",
        reset_fp=False,
        headonly=headonly,
        starttime=starttime,
        sidpattern=sidpattern,
        publication_versions=publication_versions,
        endtime=endtime,
        verbose=verbose,
    )


def _buffer_read_mseed3(
    handle: io.RawIOBase,
    headonly: bool,
    starttime: typing.Optional[obspy.UTCDateTime],
    endtime: typing.Optional[obspy.UTCDateTime],
    sidpattern: typing.Optional[str],
    publication_versions: typing.Optional[typing.Union[int, typing.List[int]]],
    verbose: typing.Union[bool, int],
) -> obspy.Stream:
    r = utils._lib.mstl3_init(C.c_void_p())

    buffer = np.fromfile(handle, dtype=np.uint8)

    flags = utils._MSF_SKIPNOTDATA
    if not headonly:
        flags |= utils._MSF_UNPACKDATA

    # Set-up tolerance callbacks.
    # XXX: Once libmseed3 has left pre-release mode this should be written in C
    # for performance reasons.

    def _time_tolerance_py(record):
        """
        Must return a time tolerance in seconds.
        """
        sr = record.contents.samprate
        # Return half a sample for now.
        return (1.0 / sr) * 0.5

    def _samprate_tolerance_py(record):
        """
        Must return a sampling rate tolerance in Hertz.
        """
        sr = record.contents.samprate
        # Return 0.1 percent of the sampling rate for now.
        return 0.001 * sr

    _time_tolerance = C.CFUNCTYPE(C.c_double, C.POINTER(utils.MS3Record))(
        _time_tolerance_py
    )
    _samprate_tolerance = C.CFUNCTYPE(C.c_double, C.POINTER(utils.MS3Record))(
        _samprate_tolerance_py
    )

    tolerance_callbacks = utils.MS3Tolerance(
        time=_time_tolerance, samprate=_samprate_tolerance
    )

    # Callback function for mstl3_pack to actually write the file
    utils._lib.mstl3_readbuffer_selection(
        C.pointer(r),
        buffer,
        buffer.size,
        # splitversion - always split!
        1,
        # flags
        flags,
        # tolerance flags.
        C.pointer(tolerance_callbacks),
        # selections.
        _assemble_selections(
            starttime=starttime,
            endtime=endtime,
            publication_versions=publication_versions,
            sidpattern=sidpattern,
        ),
        # verbose
        utils._verbosity_to_int(verbose),
    )

    # Might be empty for example when the selections return nothing.
    if r.contents.numtraces == 0:
        msg = "No data in file with the given selection."
        raise ValueError(msg)

    st = _tracelist_to_stream(r, headonly=headonly)
    utils._lib.mstl3_free(C.pointer(r), 0)
    return st


def _assemble_selections(
    *,
    starttime: typing.Optional[obspy.UTCDateTime],
    endtime: typing.Optional[obspy.UTCDateTime],
    sidpattern: typing.Optional[str],
    publication_versions: typing.Optional[typing.Union[int, typing.List[int]]],
) -> typing.Union[None, utils.MS3Selections]:
    # No selection if nothing to be selected.
    if (
        starttime is None
        and endtime is None
        and sidpattern is None
        and not publication_versions
    ):
        return None
    else:
        if isinstance(publication_versions, int):
            publication_versions = [publication_versions]

        selections = utils.MS3Selections()

        # Deal with the times.
        if starttime is not None or endtime is not None:
            time_selection = utils.MS3SelectTime()
            if starttime is not None:
                if not isinstance(starttime, obspy.UTCDateTime):
                    msg = "starttime needs to be a UTCDateTime object"
                    raise TypeError(msg)
                time_selection.starttime = starttime.ns
            else:
                time_selection.starttime = np.iinfo(np.int64).min
            if endtime is not None:
                if not isinstance(endtime, obspy.UTCDateTime):
                    msg = "endtime needs to be a UTCDateTime object"
                    raise TypeError(msg)
                time_selection.endtime = endtime.ns
            else:
                time_selection.endtime = np.iinfo(np.int64).max
            selections.timewindows.contents = time_selection

        # Now the SID pattern.
        if sidpattern is not None:
            selections.sidpattern = sidpattern.encode()
        else:
            selections.sidpattern = b"*"

        if publication_versions:
            pub_ver = copy.deepcopy(publication_versions)
            selections.pubversion = pub_ver.pop(0)
            # We need a new selections struct for every publication version.
            current_selection = selections
            for p in pub_ver:
                # Everything has to be repeated as only that will constitute a
                # valid selection.
                s = utils.MS3Selections(
                    sidpattern=selections.sidpattern, pubversion=p
                )
                if starttime is not None or endtime is not None:
                    t = utils.MS3SelectTime(
                        starttime=selections.timewindows.contents.starttime,
                        endtime=selections.timewindows.contents.endtime,
                    )
                    s.timewindows.contents = t
                current_selection.next.contents = s
                current_selection = s
        else:
            # All versions.
            selections.pub_ver = 0

        return selections


def _tracelist_to_stream(t_l, headonly: bool):
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
                    headonly=headonly,
                )
            )
            current_segment = current_segment.contents.next
        current_trace = current_trace.contents.next
    return st


def _trace_segment_to_trace(
    t_s, source_identifier: str, publication_version: int, headonly: bool
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

    # Headonly.
    if t_s.sampletype == b"\x00":
        # Sanity check.
        if t_s.numsamples != 0:
            msg = "Unpacked samples but could not determine sample type."
            raise ValueError(msg)
        tr.data = np.array([])
        tr.stats.npts = t_s.samplecnt
        return tr

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
    verbose: typing.Union[bool, int] = False,
) -> None:
    """
    :param max_record_length: Maximum record length.
    :param publication_version: Publication version for all traces if given.
        Will overwrite any per-trace settings.
    :param encoding: Data encoding. Must be compatible with the underlying
        dtype. If not given it will be chosen automatically. Int32 data will
        default to STEIM2 encoding.
    :param verbose: Controls verbosity - passed to `libmseed`.
    """
    # Map encoding string to enumerated value.
    if isinstance(encoding, str):
        encoding = utils.Encoding[encoding.upper()]

    # Don't even bother passing on the extra kwargs - this should really be
    # cleaned up on ObsPy's side.
    return _buffer_proxy(
        filename_or_buf=filename,
        function=_buffer_write_mseed3,
        file_mode="wb",
        reset_fp=False,
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
    verbose: typing.Union[bool, int] = False,
) -> None:
    # Check all encodings - this is redundant but will raise an error if
    # encoding or dtypes are invalid/incompatible without having written
    # anything beforehand.
    [
        utils._get_or_check_encoding(data=tr.data, encoding=encoding)
        for tr in stream
    ]

    # Write each trace separately.
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
    verbose: typing.Union[bool, int] = False,
) -> None:

    encoding = utils._get_or_check_encoding(data=trace.data, encoding=encoding)

    # The only case in which we'll convert data types if for int16.
    if trace.data.dtype == np.int16:
        trace = trace.copy()
        trace.data = np.require(trace.data, dtype=np.int32)

    ms_record = _trace_to_ms_record(
        trace=trace,
        publication_version=publication_version,
        record_length=max_record_length,
        encoding=encoding,
    )

    packed_samples = C.c_longlong(0)

    # Callback function for mstl3_pack to actually write the file
    def record_handler(record, reclen, _stream):
        handle.write(record[0:reclen])

    # Define Python callback function for use in C function
    rec_handler = C.CFUNCTYPE(None, C.POINTER(C.c_char), C.c_int, C.c_void_p)(
        record_handler
    )

    utils._lib.msr3_pack(
        C.pointer(ms_record),
        rec_handler,
        # Pointer passed to the callback function.
        C.c_void_p(),
        # The number of packed samples - returned to the caller.
        C.pointer(packed_samples),
        # flags. Always flush the data - seems to be what we want in ObsPy.
        utils._MSF_FLUSHDATA,
        # verbose,
        utils._verbosity_to_int(verbose),
    )

    # Assure all samples have been packed.
    if packed_samples.value != trace.stats.npts:
        msg = (
            f"Only {packed_samples.value} samples out of "
            f"{trace.stats.npts} samples have been packed."
        )
        raise ValueError(msg)


def _trace_to_ms_record(
    trace: obspy.Trace,
    record_length: int,
    encoding: utils.Encoding,
    publication_version: typing.Optional[int] = None,
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

    rec = utils.MS3Record(
        record=C.c_char_p(),
        reclen=record_length,
        # Nothing requires swapping.
        swapflag=0,
        sid=utils._nslc_to_source_id(
            network=trace.stats.network,
            station=trace.stats.station,
            location=trace.stats.location,
            channel=trace.stats.channel,
        ).encode(),
        # XXX: Is this used during writing?
        formatversion=3,
        # Record-level bit flags.
        flags=0,
        starttime=trace.stats.starttime.ns,
        # xxx: documented as "nominal sample rate as samples/second (hz) or
        # period (s)"
        #
        # This is not the same. I guess its the sampling rate in Hz?
        samprate=trace.stats.sampling_rate,
        encoding=encoding.value,
        pubversion=pub_ver,
        samplecnt=trace.stats.npts,
        # Should not matter during writing.
        crc=0,
        extralength=0,
        datalength=0,
        extra=C.c_char_p(),
        datasamples=trace.data.ctypes.data_as(C.POINTER(C.c_uint8)),
        numsamples=trace.stats.npts,
        sampletype=utils.INV_SAMPLE_TYPES[trace.data.dtype],
    )

    return rec
