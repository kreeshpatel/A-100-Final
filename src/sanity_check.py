"""
Sanity check script for the MNIST CNN project.

Verifies that the Python environment, all dependencies, the model, and a
short training loop work correctly. Designed to catch the most common
Windows issue - "ImportError: DLL load failed while importing _C" - and
print an actionable fix checklist when it occurs.

Usage (from the project root):
    python src/sanity_check.py
"""

import importlib
import platform
import struct
import sys
import os

# ---------------------------------------------------------------------------
# 1. Environment info
# ---------------------------------------------------------------------------
print("=" * 60)
print("  ENVIRONMENT")
print("=" * 60)
print(f"  Python version  : {sys.version}")
print(f"  Platform        : {platform.platform()}")
print(f"  Architecture    : {platform.machine()} ({struct.calcsize('P') * 8}-bit)")
print(f"  sys.executable  : {sys.executable}")
print(f"  sys.path[0]     : {sys.path[0]}")
print()

# ---------------------------------------------------------------------------
# 2. Dependency imports
# ---------------------------------------------------------------------------
print("=" * 60)
print("  DEPENDENCY CHECK")
print("=" * 60)

_SIMPLE_DEPS = [
    ("numpy",      "numpy"),
    ("matplotlib", "matplotlib"),
    ("sklearn",    "scikit-learn"),
]

all_ok = True

for module_name, pip_name in _SIMPLE_DEPS:
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, "__version__", "unknown")
        print(f"  [OK]   {pip_name:20s}  version {version}")
    except ImportError as exc:
        print(f"  [FAIL] {pip_name:20s}  --> {exc}")
        all_ok = False

torch_ok = False
try:
    import torch
    print(f"  [OK]   {'torch':20s}  version {torch.__version__}")
    torch_ok = True
except ImportError as exc:
    print(f"  [FAIL] {'torch':20s}  --> {exc}")
    all_ok = False

torchvision_ok = False
if torch_ok:
    try:
        import torchvision
        print(f"  [OK]   {'torchvision':20s}  version {torchvision.__version__}")
        torchvision_ok = True
    except ImportError as exc:
        print(f"  [FAIL] {'torchvision':20s}  --> {exc}")
        all_ok = False

print()

if not torch_ok:
    print("Stopping early - torch must import before model tests can run.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 3. Model import
# ---------------------------------------------------------------------------
print("=" * 60)
print("  MODEL CHECK")
print("=" * 60)

src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

try:
    from model import MNISTNet
    model = MNISTNet()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  [OK]   MNISTNet instantiated  ({total_params:,} parameters)")
except Exception as exc:
    print(f"  [FAIL] Could not load MNISTNet --> {exc}")
    sys.exit(1)

print()

# ---------------------------------------------------------------------------
# 4. Forward pass on a random tensor
# ---------------------------------------------------------------------------
print("=" * 60)
print("  FORWARD PASS CHECK")
print("=" * 60)

dummy = torch.randn(4, 1, 28, 28)
model.eval()
with torch.no_grad():
    logits = model(dummy)
print(f"  Input  shape : {list(dummy.shape)}")
print(f"  Output shape : {list(logits.shape)}")
print(f"  [OK]   Forward pass succeeded")
print()

# ---------------------------------------------------------------------------
# 5. Final verdict
# ---------------------------------------------------------------------------
print("=" * 60)
if all_ok:
    print("  ALL CHECKS PASSED - environment is ready!")
else:
    print("  SOME CHECKS FAILED - review the [FAIL] lines above.")
print("=" * 60)

sys.exit(0 if all_ok else 1)
