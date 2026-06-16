"""
Evaluation-observation helper: drive a Jupyter notebook cell-by-cell on a remote Colab runtime
from the terminal/agent. The Colab CLI has no native .ipynb runner — `colab exec` only sends a
file to the kernel — so this reads the notebook with nbformat, extracts a chosen code cell (or
range), and execs its source against a *persistent* named Colab session (kernel state carries
across calls). It lets an agent run notebooks/evaluation_observation.ipynb one cell at a time and
read each cell's output, doing the evaluation observation autonomously.

This script does NOT create or stop sessions — the caller runs `colab new`/`colab stop` around
it so session lifecycle stays explicit (see the colab-cli skill).

Execution environment: local machine (it only shells out to the `colab` CLI). Run as a module:
    python -m scripts.run_nb_cell notebooks/evaluation_observation.ipynb --list
    python -m scripts.run_nb_cell notebooks/evaluation_observation.ipynb --session amlk-eval-obs --range 1:7
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import nbformat


def code_cells(nb_path: str) -> list[str]:
    """The source of every code cell, in notebook order (markdown cells are skipped)."""
    nb = nbformat.read(nb_path, as_version=4)
    return [c.source for c in nb.cells if c.cell_type == "code"]


def list_cells(cells: list[str]) -> None:
    """Print each code cell's index and first non-empty line, so a caller can pick cells."""
    for i, src in enumerate(cells):
        first = next((ln for ln in src.splitlines() if ln.strip()), "(empty)")
        print(f"[{i}] {first[:100]}")


def exec_cell(src: str, session: str, auth: str, timeout: int) -> int:
    """Send one cell's source to the remote kernel via `colab exec -f`, streaming its output.

    Notebook magics (`!cmd`, `%pip`) are kept as-is — the Colab kernel understands them, so the
    cells run identically to the Colab UI. Returns the colab CLI exit code.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(src)
        tmp = f.name
    try:
        cmd = ["colab", f"--auth={auth}", "exec", "-s", session, "-f", tmp,
               "--timeout", str(timeout)]
        print(f"$ {' '.join(cmd)}", flush=True)
        return subprocess.run(cmd).returncode
    finally:
        Path(tmp).unlink(missing_ok=True)


def parse_selection(cells: list[str], cell: int | None, rng: str | None, run_all: bool) -> list[int]:
    """Turn --cell / --range A:B / --all into an ordered list of cell indices."""
    if run_all:
        return list(range(len(cells)))
    if rng is not None:
        a, b = rng.split(":")
        return list(range(int(a or 0), int(b) if b else len(cells)))
    if cell is not None:
        return [cell]
    return []


def main():
    parser = argparse.ArgumentParser(description="Run notebook code cells on a Colab session")
    parser.add_argument("notebook")
    parser.add_argument("--list", action="store_true", help="List code cells and exit")
    parser.add_argument("--session", help="Colab session name (from `colab new -s NAME`)")
    parser.add_argument("--cell", type=int, help="Run a single code-cell index")
    parser.add_argument("--range", help="Run an index span, e.g. 1:7 (end exclusive)")
    parser.add_argument("--all", action="store_true", help="Run every code cell in order")
    parser.add_argument("--auth", default="oauth2", help="Colab CLI auth mode")
    parser.add_argument("--timeout", type=int, default=600, help="Per-cell exec timeout (s)")
    args = parser.parse_args()

    cells = code_cells(args.notebook)
    if args.list:
        list_cells(cells)
        return

    selection = parse_selection(cells, args.cell, args.range, args.all)
    if not selection:
        print("Nothing to run: pass --list, --cell N, --range A:B, or --all", file=sys.stderr)
        sys.exit(2)
    if not args.session:
        print("ERROR: --session is required to exec cells", file=sys.stderr)
        sys.exit(2)

    for i in selection:
        print(f"\n===== cell [{i}] =====", flush=True)
        rc = exec_cell(cells[i], args.session, args.auth, args.timeout)
        if rc != 0:
            print(f"cell [{i}] failed (colab exit {rc}); stopping", file=sys.stderr)
            sys.exit(rc)


if __name__ == "__main__":
    main()
