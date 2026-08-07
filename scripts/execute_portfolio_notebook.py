#!/usr/bin/env python3
"""Execute the portfolio notebook and preserve its outputs."""

import os
from pathlib import Path

# WSL-mounted Windows temp files cannot satisfy Jupyter's POSIX mode-0600
# connection-file check.  Force every runtime/cache file onto the Linux temp
# filesystem before importing Jupyter.
RUNTIME = Path("/tmp/rogii-portfolio-jupyter")
RUNTIME.mkdir(parents=True, exist_ok=True)
RUNTIME.chmod(0o700)
for name in ("TMPDIR", "TMP", "TEMP"):
    os.environ[name] = str(RUNTIME)
os.environ["JUPYTER_RUNTIME_DIR"] = str(RUNTIME / "runtime")
os.environ["IPYTHONDIR"] = str(RUNTIME / "ipython")
os.environ["MPLCONFIGDIR"] = str(RUNTIME / "matplotlib")
for directory in (RUNTIME / "runtime", RUNTIME / "ipython", RUNTIME / "matplotlib"):
    directory.mkdir(parents=True, exist_ok=True)

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "portfolio_notebook.ipynb"
target = ROOT / "portfolio_notebook_executed.ipynb"
notebook = nbformat.read(source, as_version=4)
client = NotebookClient(
    notebook,
    timeout=300,
    kernel_name="python3",
    resources={"metadata": {"path": str(ROOT)}},
    allow_errors=False,
)
client.execute()
nbformat.write(notebook, target)
print(target)
