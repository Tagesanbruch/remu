#!/usr/bin/env python3
"""Generate ONNX model analysis report for tvm-sw/onnx.
Outputs a JSON summary to stdout or writes to a file if --out is provided.
"""
import argparse
import json
import os
from collections import Counter

import onnx
from onnx import shape_inference


def get_value_info_shape(vi):
    t = vi.type.tensor_type
    if not t.HasField("shape"):
        return None
    dims = []
    for d in t.shape.dim:
        if d.HasField("dim_value"):
            dims.append(d.dim_value)
        elif d.HasField("dim_param"):
            dims.append(d.dim_param)
        else:
            dims.append("?")
    return dims


def analyze_model(path):
    model = onnx.load(path)
    try:
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        shape_error = str(e)
    else:
        shape_error = None

    graph = model.graph
    init_names = {i.name for i in graph.initializer}
    inputs = [i for i in graph.input if i.name not in init_names]
    outputs = list(graph.output)
    ops = Counter(n.op_type for n in graph.node)

    return {
        "path": path,
        "inputs": [
            {"name": i.name, "shape": get_value_info_shape(i)} for i in inputs
        ],
        "outputs": [
            {"name": o.name, "shape": get_value_info_shape(o)} for o in outputs
        ],
        "op_types": dict(ops),
        "op_type_count": len(ops),
        "node_count": len(graph.node),
        "shape_inference_error": shape_error,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", default="-")
    args = parser.parse_args()

    model_paths = []
    for dirpath, _, filenames in os.walk(args.root):
        for f in filenames:
            if f.endswith(".onnx"):
                model_paths.append(os.path.join(dirpath, f))

    model_paths.sort()
    result = {
        "root": args.root,
        "models": [analyze_model(p) for p in model_paths],
    }

    if args.out == "-":
        print(json.dumps(result, indent=2))
    else:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
