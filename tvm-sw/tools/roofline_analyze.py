#!/usr/bin/env python3
"""Roofline analysis utility for REMU NPU model logs.

Inputs:
- <model>_roofline_meta.json from compiler/remu_tvm/core.py
- REMU runtime log containing LPROF lines emitted by generated inference C code

Outputs:
- roofline_layers.csv: per-layer joined static+runtime metrics
- roofline_summary.json: aggregate numbers and peak estimates
- roofline_report.md: human-readable summary
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from typing import Dict, List, Optional

LPROF_RE = re.compile(
    r"LPROF,([0-9a-fA-Fx]+),([^,]+),([0-9a-fA-Fx]+),([0-9a-fA-Fx]+),([0-9a-fA-Fx]+),([0-9a-fA-Fx]+),([0-9a-fA-Fx]+)"
)


def _parse_counter(token: str) -> int:
    t = token.strip().lower()
    if t.startswith("0x"):
        return int(t, 16)
    if re.search(r"[a-f]", t):
        return int(t, 16)
    return int(t, 10)


def _find_single_file_by_suffix(folder: str, suffix: str) -> Optional[str]:
    cand = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(suffix)
    ]
    if not cand:
        return None
    cand.sort()
    return cand[0]


def parse_lprof(log_path: str) -> Dict[int, Dict[str, int]]:
    rows: Dict[int, Dict[str, int]] = {}
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LPROF_RE.search(line)
            if not m:
                continue
            idx = _parse_counter(m.group(1))
            rows[idx] = {
                "idx": idx,
                "op_type_runtime": m.group(2),
                "cycles": _parse_counter(m.group(3)),
                "bytes": _parse_counter(m.group(4)),
                "gemm": _parse_counter(m.group(5)),
                "act": _parse_counter(m.group(6)),
                "dma": _parse_counter(m.group(7)),
            }
    return rows


def parse_npu_summary(log_path: str) -> Dict[str, int]:
    keys = {
        "npu_active_cycles": None,
        "memory_traffic_bytes": None,
        "gemm_ops": None,
        "activation_ops": None,
        "dma_transfers": None,
    }
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            for k in list(keys.keys()):
                if k not in s:
                    continue
                m = re.search(r"(\d+)", s)
                if m:
                    keys[k] = int(m.group(1))
    return {k: int(v) for k, v in keys.items() if v is not None}


def _numel(shape: Optional[List[int]]) -> int:
    if not shape:
        return 0
    out = 1
    for d in shape:
        out *= max(int(d), 1)
    return out


def _estimate_layer_ops_bytes(layer: Dict, onnx_weights: Dict[str, object]) -> Dict[str, int]:
    op = str(layer.get("op_type", "")).lower()
    attrs = layer.get("attrs", {}) or {}
    in_shape = layer.get("input_shape", [])
    out_shape = layer.get("output_shape", [])

    input_elems = _numel(in_shape)
    output_elems = _numel(out_shape)
    input_bytes = input_elems
    output_bytes = output_elems
    weight_bytes = 0
    ops_est = 0
    kind = "other"

    if "conv2d" in op and len(out_shape) >= 4 and len(in_shape) >= 4:
        n, cin, hin, win = [int(x) for x in in_shape[:4]]
        _, cout, hout, wout = [int(x) for x in out_shape[:4]]
        groups = int(attrs.get("groups", 1))

        kernel = attrs.get("kernel_size", [1, 1])
        if isinstance(kernel, (list, tuple)) and len(kernel) >= 2:
            kh = int(kernel[0])
            kw = int(kernel[1])
        else:
            kh = 1
            kw = 1

        macs = n * cout * hout * wout * max(cin // max(groups, 1), 1) * kh * kw
        ops_est = 2 * macs

        weight_name = layer.get("weight_name")
        if weight_name and weight_name in onnx_weights:
            arr = onnx_weights[weight_name]
            weight_bytes = int(getattr(arr, "size", 0))
        else:
            weight_bytes = cout * max(cin // max(groups, 1), 1) * kh * kw

        if groups == cin and groups == cout:
            kind = "depthwise_conv"
        else:
            kind = "conv"

    elif "dense" in op or "matmul" in op or "gemm" in op:
        if len(in_shape) >= 2 and len(out_shape) >= 2:
            m = int(in_shape[0])
            k = int(in_shape[1])
            n = int(out_shape[1])
            ops_est = 2 * m * n * k
            weight_bytes = k * n
        elif len(out_shape) >= 2:
            n = int(out_shape[1])
            k = int(in_shape[-1]) if len(in_shape) > 0 else 1
            ops_est = 2 * n * k
            weight_bytes = n * k
        kind = "dense"

    elif "add" in op or "subtract" in op or "multiply" in op:
        ops_est = output_elems
        kind = "elementwise"

    elif "relu" in op or "clip" in op or "sigmoid" in op or "tanh" in op:
        ops_est = output_elems
        kind = "activation"

    elif "pool" in op:
        ops_est = output_elems
        kind = "pool"

    elif "quantize" in op or "dequantize" in op or "requantize" in op:
        ops_est = output_elems
        kind = "quant"

    bytes_est = input_bytes + output_bytes + weight_bytes
    return {
        "ops_est": int(ops_est),
        "bytes_est": int(bytes_est),
        "weight_bytes": int(weight_bytes),
        "kind": kind,
    }


def generate_meta_from_layers_json(layers_path: str, onnx_model_path: str, model_name: str) -> Dict:
    try:
        import onnx
        from onnx import numpy_helper
    except Exception as e:  # pragma: no cover
        raise RuntimeError("onnx package is required to generate meta from layers json") from e

    model = onnx.load(onnx_model_path)
    onnx_weights = {
        init.name: numpy_helper.to_array(init)
        for init in model.graph.initializer
    }

    with open(layers_path, "r", encoding="utf-8") as f:
        layers = json.load(f)

    out_layers = []
    total_ops = 0
    total_bytes = 0

    for layer in layers:
        est = _estimate_layer_ops_bytes(layer, onnx_weights)
        out_layers.append({
            "idx": int(layer.get("idx", 0)),
            "name": layer.get("name", ""),
            "op_type": layer.get("op_type", ""),
            "kind": est["kind"],
            "input_shape": layer.get("input_shape"),
            "output_shape": layer.get("output_shape"),
            "ops_est": int(est["ops_est"]),
            "bytes_est": int(est["bytes_est"]),
            "weight_bytes": int(est["weight_bytes"]),
        })
        total_ops += est["ops_est"]
        total_bytes += est["bytes_est"]

    return {
        "model_name": model_name,
        "num_layers": len(out_layers),
        "total_ops_est": int(total_ops),
        "total_bytes_est": int(total_bytes),
        "layers": out_layers,
    }


def build_roofline(meta: Dict, runtime: Dict[int, Dict[str, int]],
                   peak_ops_per_cycle: Optional[float],
                   peak_bw_bytes_per_cycle: Optional[float]) -> Dict:
    layers_out: List[Dict] = []

    for l in meta.get("layers", []):
        idx = int(l["idx"])
        r = runtime.get(idx, {})
        cycles = int(r.get("cycles", 0))
        bytes_rt = int(r.get("bytes", 0))
        ops = int(l.get("ops_est", 0))
        bytes_fallback = int(l.get("bytes_est", 1))
        bytes_used = bytes_rt if bytes_rt > 0 else max(bytes_fallback, 1)

        ai = float(ops) / float(bytes_used) if bytes_used > 0 else 0.0
        op_per_cycle = float(ops) / float(cycles) if cycles > 0 else 0.0
        bw_per_cycle = float(bytes_used) / float(cycles) if cycles > 0 else 0.0

        layers_out.append({
            "idx": idx,
            "name": l.get("name", ""),
            "op_type": l.get("op_type", ""),
            "kind": l.get("kind", "other"),
            "ops_est": ops,
            "bytes_used": bytes_used,
            "cycles": cycles,
            "op_per_cycle": op_per_cycle,
            "bw_bytes_per_cycle": bw_per_cycle,
            "ai_ops_per_byte": ai,
            "gemm": int(r.get("gemm", 0)),
            "act": int(r.get("act", 0)),
            "dma": int(r.get("dma", 0)),
        })

    valid = [x for x in layers_out if x["cycles"] > 0 and x["ops_est"] > 0]

    if peak_ops_per_cycle is None:
        peak_ops_per_cycle = max((x["op_per_cycle"] for x in valid), default=1.0)
    if peak_bw_bytes_per_cycle is None:
        peak_bw_bytes_per_cycle = max((x["bw_bytes_per_cycle"] for x in valid), default=1.0)

    peak_ops_per_cycle = max(float(peak_ops_per_cycle), 1e-9)
    peak_bw_bytes_per_cycle = max(float(peak_bw_bytes_per_cycle), 1e-9)
    knee_ai = peak_ops_per_cycle / peak_bw_bytes_per_cycle

    for x in layers_out:
        ai = x["ai_ops_per_byte"]
        roof = min(peak_ops_per_cycle, ai * peak_bw_bytes_per_cycle)
        util = (x["op_per_cycle"] / roof) if roof > 1e-12 else 0.0
        x["roof_ops_per_cycle"] = roof
        x["utilization"] = util
        x["bound_type"] = "memory-bound" if ai < knee_ai else "compute-bound"

    total_ops = sum(x["ops_est"] for x in valid)
    total_cycles = sum(x["cycles"] for x in valid)
    total_bytes = sum(x["bytes_used"] for x in valid)

    summary = {
        "model_name": meta.get("model_name", "model"),
        "layers_total": len(layers_out),
        "layers_profiled": len(valid),
        "peak_ops_per_cycle": peak_ops_per_cycle,
        "peak_bw_bytes_per_cycle": peak_bw_bytes_per_cycle,
        "knee_ai_ops_per_byte": knee_ai,
        "total_ops_est_profiled": int(total_ops),
        "total_cycles_profiled": int(total_cycles),
        "total_bytes_profiled": int(total_bytes),
        "overall_ops_per_cycle": (float(total_ops) / float(total_cycles)) if total_cycles > 0 else 0.0,
        "overall_bw_bytes_per_cycle": (float(total_bytes) / float(total_cycles)) if total_cycles > 0 else 0.0,
    }

    return {
        "summary": summary,
        "layers": layers_out,
    }


def write_outputs(result: Dict, out_dir: str, npu_summary: Dict[str, int], top_k: int) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "roofline_layers.csv")
    json_path = os.path.join(out_dir, "roofline_summary.json")
    md_path = os.path.join(out_dir, "roofline_report.md")

    fields = [
        "idx", "name", "op_type", "kind", "cycles", "ops_est", "bytes_used",
        "ai_ops_per_byte", "op_per_cycle", "bw_bytes_per_cycle",
        "roof_ops_per_cycle", "utilization", "bound_type", "gemm", "act", "dma",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in result["layers"]:
            w.writerow({k: row.get(k) for k in fields})

    json_obj = {
        "summary": result["summary"],
        "npu_summary_from_log": npu_summary,
        "layers": result["layers"],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False)

    layers_sorted = sorted(result["layers"], key=lambda x: x["cycles"], reverse=True)
    hot = layers_sorted[:top_k]

    s = result["summary"]
    lines: List[str] = []
    lines.append(f"# Roofline Analysis Report ({s['model_name']})")
    lines.append("")
    lines.append("## Aggregate Summary")
    lines.append("")
    lines.append(f"- layers_total: {s['layers_total']}")
    lines.append(f"- layers_profiled: {s['layers_profiled']}")
    lines.append(f"- peak_ops_per_cycle: {s['peak_ops_per_cycle']:.6f}")
    lines.append(f"- peak_bw_bytes_per_cycle: {s['peak_bw_bytes_per_cycle']:.6f}")
    lines.append(f"- knee_ai_ops_per_byte: {s['knee_ai_ops_per_byte']:.6f}")
    lines.append(f"- total_ops_est_profiled: {s['total_ops_est_profiled']}")
    lines.append(f"- total_cycles_profiled: {s['total_cycles_profiled']}")
    lines.append(f"- total_bytes_profiled: {s['total_bytes_profiled']}")
    lines.append(f"- overall_ops_per_cycle: {s['overall_ops_per_cycle']:.6f}")
    lines.append(f"- overall_bw_bytes_per_cycle: {s['overall_bw_bytes_per_cycle']:.6f}")
    lines.append("")

    if npu_summary:
        lines.append("## NPU Summary In Log")
        lines.append("")
        for k, v in npu_summary.items():
            lines.append(f"- {k}: {v}")
        lines.append("")

    lines.append(f"## Top-{len(hot)} Hot Layers By Cycles")
    lines.append("")
    lines.append("| idx | op_type | cycles | ops_est | AI(ops/B) | perf(ops/cycle) | roof | util | bound |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---|")
    for x in hot:
        lines.append(
            f"| {x['idx']} | {x['op_type']} | {x['cycles']} | {x['ops_est']} | "
            f"{x['ai_ops_per_byte']:.4f} | {x['op_per_cycle']:.4f} | {x['roof_ops_per_cycle']:.4f} | "
            f"{x['utilization']:.4f} | {x['bound_type']} |"
        )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return {
        "csv": csv_path,
        "json": json_path,
        "markdown": md_path,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze REMU LPROF logs with roofline model")
    p.add_argument("--model-dir", required=True, help="Model output directory, e.g. compiler/output/mobilenet")
    p.add_argument("--meta-file", default=None, help="Path to <model>_roofline_meta.json")
    p.add_argument("--layers-file", default=None, help="Path to <model>_layers.json (for fallback meta generation)")
    p.add_argument("--onnx-model", default=None, help="Path to ONNX model (for fallback meta generation)")
    p.add_argument("--model-name", default=None, help="Model name for fallback meta generation")
    p.add_argument("--log-file", default=None, help="Path to runtime log containing LPROF lines")
    p.add_argument("--out-dir", default=None, help="Output directory for analysis artifacts")
    p.add_argument("--peak-ops-per-cycle", type=float, default=None, help="Override compute roof peak")
    p.add_argument("--peak-bw-bytes-per-cycle", type=float, default=None, help="Override memory roof peak")
    p.add_argument("--top-k", type=int, default=15, help="Top-k layers in report")
    args = p.parse_args()

    model_dir = os.path.abspath(args.model_dir)

    if args.meta_file:
        meta_path = args.meta_file
    else:
        meta_path = _find_single_file_by_suffix(model_dir, "_roofline_meta.json")

    if args.log_file:
        log_path = args.log_file
    else:
        default_log = os.path.join(model_dir, "build", "remu-log.txt")
        if os.path.exists(default_log):
            log_path = default_log
        else:
            fallback = os.path.join(model_dir, "remu_run.log")
            if os.path.exists(fallback):
                log_path = fallback
            else:
                raise FileNotFoundError("runtime log not found; pass --log-file")

    out_dir = args.out_dir or os.path.join(model_dir, "roofline")

    if meta_path and os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    else:
        if not args.layers_file or not args.onnx_model:
            raise FileNotFoundError(
                "roofline meta file not found; provide --meta-file or (--layers-file and --onnx-model)"
            )

        model_name = args.model_name or os.path.basename(model_dir.rstrip("/"))
        meta = generate_meta_from_layers_json(args.layers_file, args.onnx_model, model_name)
        if not meta_path:
            meta_path = os.path.join(model_dir, f"{model_name}_roofline_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    runtime = parse_lprof(log_path)
    npu_summary = parse_npu_summary(log_path)

    result = build_roofline(
        meta=meta,
        runtime=runtime,
        peak_ops_per_cycle=args.peak_ops_per_cycle,
        peak_bw_bytes_per_cycle=args.peak_bw_bytes_per_cycle,
    )
    paths = write_outputs(result, out_dir, npu_summary, top_k=max(args.top_k, 1))

    print("Roofline analysis complete")
    print(f"  Meta file: {meta_path}")
    print(f"  Log file: {log_path}")
    print(f"  CSV: {paths['csv']}")
    print(f"  JSON: {paths['json']}")
    print(f"  Markdown: {paths['markdown']}")


if __name__ == "__main__":
    main()
