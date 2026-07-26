"""Work around a broken local Python build that is missing the `_lzma` C extension.

`data_curation.data_download.download_hesum` needs `datasets.load_dataset`, and importing
`datasets` transitively imports the stdlib `lzma` module even though HeSum ships as plain
(non-xz) parquet. On this machine's pyenv build, `_lzma` was never compiled (a pre-existing,
already-documented environment issue — see AGENTS.md), so `import datasets` fails before any
network call is made. `ensure_lzma_importable()` installs a process-local stub `_lzma` module
that satisfies `lzma.py`'s `from _lzma import *`, without ever performing real xz compression;
call it once, before the first `import datasets` or `import lzma`, in any script that needs to
run locally without a rebuilt Python.
"""

from __future__ import annotations

import sys
import types


def ensure_lzma_importable() -> None:
    """Make `import lzma` succeed even when the `_lzma` C extension is missing."""
    try:
        import lzma  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    if "_lzma" in sys.modules:
        return

    stub = types.ModuleType("_lzma")

    class LZMAError(Exception):
        pass

    def _unsupported(*_args, **_kwargs):
        raise LZMAError("lzma is not available in this environment (stubbed _lzma module)")

    stub.LZMAError = LZMAError
    stub.LZMACompressor = _unsupported
    stub.LZMADecompressor = _unsupported
    stub._encode_filter_properties = _unsupported
    stub._decode_filter_properties = _unsupported
    stub.is_check_supported = lambda _check: False

    # Constants `lzma.py` pulls in via `from _lzma import *`; values only need to be valid
    # Python objects, since the compressor/decompressor paths that would use them are stubbed.
    constants = {
        "CHECK_NONE": 0, "CHECK_CRC32": 1, "CHECK_CRC64": 4, "CHECK_SHA256": 10,
        "CHECK_ID_MAX": 15, "CHECK_UNKNOWN": 16,
        "FILTER_LZMA1": 0x4000000000000001, "FILTER_LZMA2": 0x21,
        "FILTER_DELTA": 0x03, "FILTER_X86": 0x04, "FILTER_IA64": 0x06,
        "FILTER_ARM": 0x07, "FILTER_ARMTHUMB": 0x08, "FILTER_POWERPC": 0x05,
        "FILTER_SPARC": 0x09,
        "FORMAT_AUTO": 0, "FORMAT_XZ": 1, "FORMAT_ALONE": 2, "FORMAT_RAW": 3,
        "MF_HC3": 0x03, "MF_HC4": 0x04, "MF_BT2": 0x12, "MF_BT3": 0x13, "MF_BT4": 0x14,
        "MODE_FAST": 1, "MODE_NORMAL": 2,
        "PRESET_DEFAULT": 6, "PRESET_EXTREME": 1 << 31,
    }
    for name, value in constants.items():
        setattr(stub, name, value)

    sys.modules["_lzma"] = stub
