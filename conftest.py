"""
Root pytest configuration.
Ensures Windows DLL directory for PyTorch is registered early to prevent WinError 1114.
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path and initialize Windows PyTorch DLLs
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.utils.torch_loader  # noqa: F401
