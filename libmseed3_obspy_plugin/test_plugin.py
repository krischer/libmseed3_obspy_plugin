import itertools
import pathlib

import pytest

from libmseed3_obspy_plugin.core import read, _is_mseed

data = pathlib.Path(__file__).parent / "src" / "libmseed" / "test"


@pytest.mark.parametrize(
    "path, expected_result",
    # All the pack- files are MSEED3 files.
    list(zip(data.glob("pack-*-encoded.test.ref"), itertools.repeat(True))) +
    # All others are currently MSEED2 or some other files.
    list(zip(data.glob("read-*"), itertools.repeat(False))),
)
def test_is_mseed(path, expected_result):
    assert _is_mseed(path) is expected_result


def test_read_by_unpacking_libmseed3_packed_data():
    """
    libmseed3 currently comes with a couple of MiniSEED3 files that it
    creates from reference data. We'll just test reading these here.
    """