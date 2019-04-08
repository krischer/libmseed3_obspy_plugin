import pathlib
from libmseed3_obspy_plugin.core import read

data = pathlib.Path(__file__).parent / "src" / "libmseed" / "test"


def test_something():
    filename = data / "pack-Float64-encoded.test.ref"
    st = read(filename)
