"""Run every test: rules and maths, solver, and one regression test per bug found so far.

    .venv/bin/python test.py           # everything, including a full planner run (~30 s)
    .venv/bin/python test.py --quick   # skip the slow end-to-end planner run
"""
import os
import subprocess
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
args = [sys.executable, "-m", "pytest", "tests", "-q"]
if "--quick" in sys.argv:
    args += ["-m", "not slow"]
sys.exit(subprocess.call(args))
