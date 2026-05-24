#!/usr/bin/env python3
"""Sanitize notebooks in place so nbconvert --to html won't crash on them.

Two failure modes this fixes (both un-renderable in static HTML anyway, so no
visible content is lost):
  1. ipywidgets state: a `metadata.widgets` / widget-state+json block without a
     "state" key makes nbconvert's widget filter raise KeyError: 'state'.
     We drop `widgets` from notebook-, cell-, and output-level metadata.
  2. invalid nbformat: a `stream` output carrying a `metadata` property is
     rejected ("Additional properties are not allowed ('metadata' …)").
     We drop `metadata` from stream outputs.

Usage: sanitize_notebooks.py <dir>   # rewrites *.ipynb under <dir> in place
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def sanitize(nb: dict) -> bool:
    changed = False

    def drop_widgets(meta: dict) -> None:
        nonlocal changed
        if isinstance(meta, dict) and "widgets" in meta:
            del meta["widgets"]
            changed = True

    drop_widgets(nb.get("metadata", {}))
    for cell in nb.get("cells", []):
        if not isinstance(cell, dict):
            continue
        drop_widgets(cell.get("metadata", {}))
        for out in cell.get("outputs", []):
            if not isinstance(out, dict):
                continue
            drop_widgets(out.get("metadata", {}))
            # stream outputs may not carry a `metadata` property at all
            if out.get("output_type") == "stream" and "metadata" in out:
                del out["metadata"]
                changed = True
    return changed


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: sanitize_notebooks.py <dir>", file=sys.stderr)
        return 2
    root = Path(sys.argv[1])
    n_changed = 0
    for p in root.rglob("*.ipynb"):
        if ".ipynb_checkpoints" in p.parts:
            continue
        try:
            nb = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"skip (unparseable) {p}: {e}", file=sys.stderr)
            continue
        if sanitize(nb):
            p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
            n_changed += 1
            print(f"sanitized {p}")
    print(f"sanitized {n_changed} notebook(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
