#!/usr/bin/env python3
"""Inject per-layer LPROF hooks into generated *_inference.c files.

This is a transitional helper for already-generated artifacts when recompiling from TVM
is temporarily unavailable in the current environment.
"""

from __future__ import annotations

import argparse
import os
import re
from typing import List, Optional, Tuple

LAYER_RE = re.compile(r"^(\s*)// === Layer\s+(\d+):\s*(.*?)\s*===\s*$")
SIG_RE = re.compile(r"\bint\s+\w+_inference\s*\(")


def _emit_layer_begin(indent: str, idx: int) -> List[str]:
    return [
        f"{indent}#if NPU_PROFILE_LAYERS\n",
        f"{indent}uint32_t __lp_cyc_beg_{idx} = npu_get_cycles();\n",
        f"{indent}uint32_t __lp_mem_beg_{idx} = npu_get_mem_bytes();\n",
        f"{indent}uint32_t __lp_gemm_beg_{idx} = npu_get_gemm_count();\n",
        f"{indent}uint32_t __lp_act_beg_{idx} = npu_get_act_count();\n",
        f"{indent}uint32_t __lp_dma_beg_{idx} = npu_get_dma_count();\n",
        f"{indent}#endif\n",
    ]


def _emit_layer_end(indent: str, idx: int, op: str) -> List[str]:
    op = op.replace('"', "'").replace(",", "_")
    return [
        f"{indent}#if NPU_PROFILE_LAYERS\n",
        f"{indent}printf(\"LPROF,{idx},{op},%u,%u,%u,%u,%u\\n\",\n",
        f"{indent}       (unsigned)(npu_get_cycles() - __lp_cyc_beg_{idx}),\n",
        f"{indent}       (unsigned)(npu_get_mem_bytes() - __lp_mem_beg_{idx}),\n",
        f"{indent}       (unsigned)(npu_get_gemm_count() - __lp_gemm_beg_{idx}),\n",
        f"{indent}       (unsigned)(npu_get_act_count() - __lp_act_beg_{idx}),\n",
        f"{indent}       (unsigned)(npu_get_dma_count() - __lp_dma_beg_{idx}));\n",
        f"{indent}#endif\n",
    ]


def _ensure_macro_block(lines: List[str]) -> List[str]:
    text = "".join(lines)
    if "#ifndef NPU_PROFILE_LAYERS" in text:
        return lines

    insert_at = -1
    for i, line in enumerate(lines):
        if "_weights.h" in line and line.strip().startswith("#include"):
            insert_at = i + 1
            break

    if insert_at < 0:
        insert_at = 0

    block = [
        "\n",
        "#ifndef NPU_PROFILE_LAYERS\n",
        "#define NPU_PROFILE_LAYERS 0\n",
        "#endif\n",
    ]
    return lines[:insert_at] + block + lines[insert_at:]


def _strip_existing_layer_instrumentation(lines: List[str]) -> List[str]:
    out: List[str] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        s = line.strip()

        # Remove previously injected single lines.
        if "__lp_cyc_beg_" in line or "__lp_mem_beg_" in line or "__lp_gemm_beg_" in line or "__lp_act_beg_" in line or "__lp_dma_beg_" in line:
            i += 1
            continue
        if "LPROF," in line:
            i += 1
            continue

        # Remove previously injected profiling blocks in function body.
        if s == "#if NPU_PROFILE_LAYERS" and i + 1 < n:
            look = "".join(lines[i + 1:i + 10])
            if "__lp_cyc_beg_" in look or "LPROF," in look:
                i += 1
                while i < n and lines[i].strip() != "#endif":
                    i += 1
                if i < n and lines[i].strip() == "#endif":
                    i += 1
                continue

        out.append(line)
        i += 1

    return out


def instrument_file(path: str) -> Tuple[bool, str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    lines = _ensure_macro_block(lines)
    lines = _strip_existing_layer_instrumentation(lines)

    in_infer = False
    brace_depth = 0
    pending_layer: Optional[Tuple[str, int, str]] = None
    out: List[str] = []
    changed = False

    for line in lines:
        if not in_infer and SIG_RE.search(line):
            in_infer = True
            brace_depth = line.count("{") - line.count("}")
            if brace_depth <= 0:
                brace_depth = 1

        if in_infer and pending_layer is not None:
            if "// Copy final output" in line or (brace_depth == 1 and line.strip() == "}"):
                ind, idx, op = pending_layer
                out.extend(_emit_layer_end(ind, idx, op))
                pending_layer = None
                changed = True

        m = LAYER_RE.match(line)
        if in_infer and m:
            indent = m.group(1)
            idx = int(m.group(2))
            op = m.group(3).strip()

            if pending_layer is not None:
                pind, pidx, pop = pending_layer
                out.extend(_emit_layer_end(pind, pidx, pop))
                changed = True

            out.append(line)
            out.extend(_emit_layer_begin(indent, idx))
            pending_layer = (indent, idx, op)
            changed = True
            continue

        out.append(line)

        if in_infer:
            brace_depth += line.count("{") - line.count("}")
            if brace_depth <= 0:
                in_infer = False
                pending_layer = None

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(out)

    if changed:
        return True, "instrumented"
    return False, "no layer markers found"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inject LPROF hooks into generated inference C files")
    parser.add_argument("files", nargs="+", help="Paths to *_inference.c files")
    args = parser.parse_args()

    for fp in args.files:
        p = os.path.abspath(fp)
        changed, reason = instrument_file(p)
        flag = "UPDATED" if changed else "SKIPPED"
        print(f"[{flag}] {p}: {reason}")


if __name__ == "__main__":
    main()
