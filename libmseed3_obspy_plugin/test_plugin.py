import io
import itertools
import pathlib

import numpy as np
import obspy
import pytest

from libmseed3_obspy_plugin.core import _is_mseed3
from libmseed3_obspy_plugin import utils

data = pathlib.Path(__file__).parent / "src" / "libmseed" / "test"


@pytest.mark.parametrize(
    "sid, net, sta, loc, chan",
    [
        ("XFDSN:NET_STA_LOC_B_S_P", "NET", "STA", "LOC", "BSP"),
        ("XFDSN:AGENCY:NET_STA_LOC_B_S_P", "NET", "STA", "LOC", "BSP"),
        ("XFDSN:XX_TEST__L_H_Z", "XX", "TEST", "", "LHZ"),
        ("XFDSN:___", "", "", "", ""),
        ("XFDSN:_AA__", "", "AA", "", ""),
        ("XFDSN:_AA__Z", "", "AA", "", "Z"),
        ("XFDSN:_AA__B_H_Z", "", "AA", "", "BHZ"),
    ],
)
def test_source_id_to_nslc(sid, net, sta, loc, chan):
    assert utils._source_id_to_nslc(sid=sid) == (net, sta, loc, chan)


@pytest.mark.parametrize(
    "net, sta, loc, chan, sid",
    [
        ("NET", "STA", "LOC", "BSP", "XFDSN:NET_STA_LOC_B_S_P"),
        ("XX", "TEST", "", "LHZ", "XFDSN:XX_TEST__L_H_Z"),
        ("XX", "TEST", "", "RANDOM", "XFDSN:XX_TEST__RANDOM"),
        ("", "", "", "", "XFDSN:___"),
        ("", "AA", "", "", "XFDSN:_AA__"),
        ("", "AA", "", "Z", "XFDSN:_AA__Z"),
        ("", "AA", "", "BHZ", "XFDSN:_AA__B_H_Z"),
    ],
)
def test_nslc_to_source_id(net, sta, loc, chan, sid):
    assert (
        utils._nslc_to_source_id(
            network=net, station=sta, location=loc, channel=chan
        )
        == sid
    )


@pytest.mark.parametrize(
    "path, expected_result",
    # All the pack- files are MSEED3 files.
    list(zip(data.glob("pack-*-encoded.test.ref"), itertools.repeat(True))) +
    # All others are currently MSEED2 or some other files.
    list(zip(data.glob("read-*"), itertools.repeat(False))),
)
def test_is_mseed(path, expected_result):
    assert _is_mseed3(path) is expected_result


@pytest.mark.parametrize(
    "path, dtype",
    [
        (data / "pack-Float32-encoded.test.ref", np.float32),
        (data / "pack-Float64-encoded.test.ref", np.float64),
        (data / "pack-Int16-encoded.test.ref", np.int32),
        (data / "pack-Int32-encoded.test.ref", np.int32),
        (data / "pack-Steim1-encoded.test.ref", np.int32),
        (data / "pack-Steim2-encoded.test.ref", np.int32),
        (data / "pack-text-encoded.test.ref", np.dtype("|S1")),
    ],
)
def test_read_by_unpacking_libmseed3_packed_data(path, dtype):
    """
    libmseed3 currently comes with a couple of MiniSEED3 files that it
    creates from reference data. We'll just test reading these here.
    """
    # fmt: off
    numeric_data = np.array([
        0, 2, 4, 5, 7, 9, 10, 11, 11, 11, 11, 10, 8, 6, 4, 1, 0, -3, -6, -8,
        -11, -13, -14, -15, -16, -15, -14, -13, -11, -8, -5, -1, 2, 6, 9, 13,
        16, 18, 20, 21, 22, 21, 19, 17, 14, 10, 5, 0, -4, -9, -14, -19, -23,
        -26, -29, -30, -30, -29, -26, -22, -18, -12, -5, 1, 8, 15, 22, 28,
        33, 38, 40, 41, 41, 39, 35, 29, 22, 14, 5, -4, -14, -24, -33, -41,
        -48, -53, -56, -57, -56, -52, -46, -38, -27, -16, -3, 10, 23, 37, 49,
        60, 68, 75, 78, 78, 75, 69, 60, 48, 33, 17, 0, -19, -38, -56, -72,
        -86, -97, -104, -108, -107, -102, -92, -78, -60, -39, -16, 8, 34, 59,
        83, 105, 123, 137, 146, 149, 146, 137, 122, 101, 75, 45, 12, -22,
        -57, -92, -124, -152, -175, -192, -202, -204, -198, -183, -160, -129,
        -92, -50, -3, 44, 93, 139, 182, 219, 249, 269, 280, 279, 267, 243,
        208, 164, 110, 50, -13, -80, -146, -209, -266, -314, -352, -376,
        -386, -380, -359, -322, -270, -205, -128, -44, 45, 137, 227, 311,
        386, 449, 495, 523, 530, 516, 480, 423, 346, 252, 144, 25, -99, -225,
        -347, -460, -558, -637, -694, -724, -726, -698, -640, -553, -440,
        -305, -151, 15, 187, 359, 524, 673, 801, 902, 969, 999, 990, 939,
        848, 718, 554, 359, 142, -89, -327, -561, -782, -980, -1145, -1271,
        -1349, -1375, -1346, -1260, -1118, -925, -686, -409, -104, 218, 544,
        862, 1157, 1417, 1629, 1784, 1871, 1885, 1822, 1681, 1466, 1181, 836,
        443, 15, -430, -877, -1306, -1700, -2039, -2309, -2495, -2586, -2575,
        -2457, -2233, -1909, -1492, -997, -441, 156, 771, 1381, 1958, 2479,
        2920, 3259, 3478, 3562, 3504, 3300, 2951, 2467, 1861, 1154, 371,
        -460, -1306, -2134, -2908, -3595, -4162, -4582, -4830, -4890, -4752,
        -4413, -3878, -3162, -2286, -1281, -181, 971, 2131, 3252, 4285, 5184,
        5908, 6418, 6686, 6690, 6419, 5874, 5064, 4013, 2753, 1329, -208,
        -1801, -3385, -4896, -6268, -7438, -8351, -8959, -9223, -9120, -8637,
        -7779, -6566, -5034, -3231, -1221, 921, 3114, 5270, 7298, 9110,
        10622, 11760, 12462, 12680, 12386, 11571, 10247, 8447, 6225, 3657,
        832, -2143, -5153, -8076, -10787, -13167, -15103, -16499, -17274,
        -17372, -16759, -15432, -13417, -10766, -7564, -3920, 36, 4153, 8270,
        12217, 15825, 18930, 21386, 23065, 23864, 23716, 22587, 20483, 17450,
        13576, 8983, 3832, -1687, -7367, -12977, -18286, -23061, -27086,
        -30164, -32129, -32855, -32260, -30315, -27044, -22526, -16897,
        -10341, -3089, 4587, 12392, 20010, 27120, 33408, 38583, 42385, 44602,
        45078, 43720, 40510, 35502, 28830, 20697, 11378, 1207, -9432, -20123,
        -30427, -39906, -48136, -54727, -59340, -61705, -61632, -59023,
        -53880, -46311, -36527, -24838, -11646, 2567, 17262, 31853, 45737,
        58314, 69012, 77316, 82784, 85076, 83966, 79360, 71301, 59981, 45728,
        29009, 10408, -9387, -29613, -49456, -68085, -84684, -98487, -108811,
        -115090, -116897, -113975, -106249, -93837, -77057, -56418, -32610,
        -6480, 20993, 48737, 75622, 100508, 122289, 139942, 152573, 159453,
        160065, 154123, 141602, 122746, 98068, 68342, 34581, -1992, -39992,
        -77915, -114200, -147287, -175686, -198036, -213168, -220164,
        -218401, -207588, -187799, -159476], dtype=np.int32)
    # fmt: on

    text_data = np.frombuffer(
        "I've seen things you people wouldn't believe. Attack ships on fire "
        "off the shoulder of Orion. I watched C-beams glitter in the dark "
        "near the Tannhäuser Gate. All those moments will be lost in time, "
        "like tears...in...rain. Time to die.".encode(),
        dtype=np.dtype("|S1"),
    )

    if dtype in (np.int32, np.float32, np.float64):
        data = np.require(numeric_data, dtype=dtype)
    elif dtype == np.dtype("|S1"):
        data = text_data
    else:  # pragma: no cover
        raise NotImplementedError

    # Only the first 400 entries can be represented by 16bit.
    if "Int16" in str(path.stem):
        data = data[:400]

    st = obspy.read(str(path))

    assert len(st) == 1
    np.testing.assert_equal(st[0].data, data)
    assert st[0].stats.network == "XX"
    assert st[0].stats.station == "TEST"
    assert st[0].stats.location == ""
    assert st[0].stats.channel == "LHZ"
    assert st[0].stats.starttime == obspy.UTCDateTime(2012, 1, 1)
    assert st[0].stats.sampling_rate == 1.0
    assert st[0].data.dtype == dtype
    assert st[0].stats.mseed3.source_identifier == "XFDSN:XX_TEST__L_H_Z"
    assert st[0].stats.mseed3.publication_version == 1


@pytest.mark.parametrize(
    "dtype, encoding",
    [
        [np.int32, utils.Encoding.STEIM1],
        [np.int32, utils.Encoding.STEIM2],
        [np.int32, utils.Encoding.INT32],
        [np.int16, utils.Encoding.INT16],
        [np.float32, utils.Encoding.FLOAT32],
        [np.float64, utils.Encoding.FLOAT64],
        [np.dtype("|S1"), utils.Encoding.ASCII],
    ],
)
def test_write_read_roundtripping(dtype, encoding):
    if encoding == utils.Encoding.ASCII:
        data = np.frombuffer(
            "I've seen things you people wouldn't believe. Attack ships on "
            "fire off the shoulder of Orion. I watched C-beams glitter in the "
            "dark near the Tannhäuser Gate. All those moments will be lost in "
            "time, like tears...in...rain. Time to die.".encode(),
            dtype=np.dtype("|S1"),
        )
    else:
        data = np.arange(500, dtype=dtype)

    tr = obspy.Trace(data=data)
    tr.stats.network = "AA"
    tr.stats.station = "BB"
    tr.stats.location = "CC"
    tr.stats.channel = "BHZ"
    tr.stats.starttime = obspy.UTCDateTime(2015, 2, 4, 2, 4, 5, 32)
    tr.stats.sampling_rate = 22.0

    with io.BytesIO() as buf:
        tr.write(buf, format="mseed3")
        buf.seek(0, 0)
        st_new = obspy.read(buf)

    assert len(st_new) == 1
    tr_new = st_new[0]

    assert tr_new.stats.network == "AA"
    assert tr_new.stats.station == "BB"
    assert tr_new.stats.location == "CC"
    assert tr_new.stats.channel == "BHZ"
    assert tr_new.stats.starttime == obspy.UTCDateTime(2015, 2, 4, 2, 4, 5, 32)
    assert tr_new.stats.sampling_rate == 22.0
    assert tr_new.stats.mseed3.source_identifier == "XFDSN:AA_BB_CC_B_H_Z"
    np.testing.assert_equal(tr.data, tr_new.data)


def test_read_write_roundtripping_different_dtypes_per_trace():
    tr1 = obspy.Trace(
        data=np.arange(100, dtype=np.float32),
        header={"station": "AA", "sampling_rate": 11.2},
    )
    tr2 = obspy.Trace(
        data=np.arange(100, dtype=np.float64),
        header={"station": "BB", "sampling_rate": 22.2},
    )
    tr3 = obspy.Trace(
        data=np.arange(200, dtype=np.int32),
        header={"station": "CC", "sampling_rate": 0.03},
    )

    st = obspy.Stream(traces=[tr1, tr2, tr3])

    with io.BytesIO() as buf:
        st.write(buf, format="mseed3")
        buf.seek(0, 0)
        st2 = obspy.read(buf)

    # Delete mseed3 spectific attributes filled during reading.
    for tr in st2:
        del tr.stats.mseed3
        del tr.stats._format

    assert st == st2

    # Also assert the dtypes.
    for tr1, tr2 in zip(st, st2):
        assert tr1.data.dtype == tr2.data.dtype


def test_roundtripping_multi_record_file():
    tr = obspy.Trace(data=np.arange(10000, dtype=np.float64))
    with io.BytesIO() as buf:
        tr.write(buf, format="mseed3", verbose=2)
        buf.seek(0, 0)
        tr_out = obspy.read(buf, verbose=2)[0]

    del tr_out.stats.mseed3
    del tr_out.stats._format

    assert tr == tr_out
