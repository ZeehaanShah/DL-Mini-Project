"""Environment audit script for DL MiniProject."""
import sys
import platform
import os

print("=" * 60)
print("ENVIRONMENT AUDIT")
print("=" * 60)
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Platform: {platform.platform()}")
print(f"Architecture: {platform.machine()}")
print(f"OS: {platform.system()} {platform.release()}")
print()

# Check virtual environment
venv = os.environ.get("VIRTUAL_ENV")
conda = os.environ.get("CONDA_DEFAULT_ENV")
print(f"Virtual environment (VIRTUAL_ENV): {venv or 'Not detected'}")
print(f"Conda environment: {conda or 'Not detected'}")
print()

# Check libraries
libraries = [
    ("torch", "PyTorch"),
    ("torchvision", "torchvision"),
    ("pandas", "pandas"),
    ("numpy", "NumPy"),
    ("sklearn", "scikit-learn"),
    ("pyarrow", "PyArrow"),
    ("PIL", "Pillow"),
    ("matplotlib", "matplotlib"),
    ("yaml", "PyYAML"),
    ("scipy", "SciPy"),
    ("tqdm", "tqdm"),
    ("io", "io (stdlib)"),
    ("json", "json (stdlib)"),
    ("csv", "csv (stdlib)"),
]

print("Library versions:")
print("-" * 40)
for module_name, display_name in libraries:
    try:
        mod = __import__(module_name)
        version = getattr(mod, "__version__", "installed (no version attr)")
        print(f"  {display_name:20s}: {version}")
    except ImportError:
        print(f"  {display_name:20s}: NOT INSTALLED")

print()

# Check PyTorch details
try:
    import torch
    print("PyTorch details:")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU count: {torch.cuda.device_count()}")
    else:
        print("  Running on CPU only")
    print(f"  PyTorch build: {torch.__version__}")
except ImportError:
    print("PyTorch: NOT INSTALLED")

print()

# Check disk space (Windows)
try:
    import shutil
    total, used, free = shutil.disk_usage("C:\\")
    print(f"Disk space (C:): {free / (1024**3):.1f} GB free / {total / (1024**3):.1f} GB total")
except Exception as e:
    print(f"Disk space check failed: {e}")

# Check RAM (approximate)
try:
    import ctypes
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]
    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    print(f"RAM: {stat.ullAvailPhys / (1024**3):.1f} GB free / {stat.ullTotalPhys / (1024**3):.1f} GB total")
except Exception as e:
    print(f"RAM check failed: {e}")

print()

# Check git
print("Git status:")
try:
    import subprocess
    result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=os.path.dirname(__file__) or ".")
    if result.returncode == 0:
        print("  Git repository: Yes")
        if result.stdout.strip():
            lines = result.stdout.strip().split("\n")
            print(f"  Uncommitted changes: {len(lines)} files")
        else:
            print("  Working tree: clean")
    else:
        print(f"  Git repository: No (or error: {result.stderr.strip()})")
except FileNotFoundError:
    print("  Git: NOT INSTALLED or not in PATH")

print()
print("=" * 60)
print("ENVIRONMENT AUDIT COMPLETE")
print("=" * 60)
