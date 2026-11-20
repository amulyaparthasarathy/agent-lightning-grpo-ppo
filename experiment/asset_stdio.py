"""Isolated MCP child bootstrap: no inherited credentials, labels, or Python paths."""
import os
from pathlib import Path
import sys

os.environ.clear()
os.environ.update(PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
# Kill this owned child if its MCP client disappears, even outside launcher cleanup.
import ctypes
import signal
parent = os.getppid()
if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL) != 0:
    raise RuntimeError('Unable to set MCP parent-death signal')
if os.getppid() != parent or parent == 1:
    raise RuntimeError('MCP parent disappeared')
if len(sys.argv) == 2:
    from owned_processes import register
    register(sys.argv[1])
from asset_server import main
import asyncio
asyncio.run(main())
