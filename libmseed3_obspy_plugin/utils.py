import ctypes as C
import pathlib
import warnings

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
