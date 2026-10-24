"""
Windows PyTorch DLL compatibility helper.

On Windows with Python 3.8+, PyTorch 2.14.1+cpu dynamic libraries (c10.dll, torch_cpu.dll)
fail to initialize with WinError 1114 if NumPy/OpenBLAS runtime libraries are loaded first.
This module registers torch/lib via os.add_dll_directory and performs the initial torch
import before conflicting runtime libraries enter the process space.

Linux and macOS:
Completely a no-op (os.name != 'nt').
"""
import os
from pathlib import Path
from typing import List, Any

_DLL_HANDLES: List[Any] = []
_INITIALIZED: bool = False


def init_torch_dlls() -> None:
    """Initialize Windows PyTorch DLL directories and pre-load torch runtime."""
    global _INITIALIZED
    if _INITIALIZED:
        return

    # No-op on non-Windows platforms
    if os.name != "nt":
        _INITIALIZED = True
        return

    if hasattr(os, "add_dll_directory"):
        try:
            import importlib.util
            spec = importlib.util.find_spec("torch")
            if spec and spec.origin:
                torch_lib = Path(spec.origin).parent / "lib"
                if torch_lib.exists():
                    handle = os.add_dll_directory(str(torch_lib))
                    _DLL_HANDLES.append(handle)
            # Pre-load torch while DLL descriptor handle is active
            import torch  # noqa: F401
        except Exception as exc:
            # Re-raise if torch is installed but failed to load DLLs
            import importlib.util
            if importlib.util.find_spec("torch") is not None:
                raise ImportError(
                    f"Windows PyTorch initialization failed: {exc}. "
                    "Ensure MSVC C++ Redistributable 2015-2022 is installed."
                ) from exc

    _INITIALIZED = True


# Run initialization on import
init_torch_dlls()
