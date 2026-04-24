"""Audit the exported PP-DocLayoutV3 ONNX graph for batch=1 constants.

Finds every Reshape/Expand/Tile whose shape input is a Constant
initializer (or a ConstantOfShape/Concat chain that resolves to one),
and reports those whose shape starts with a literal 1 in position 0 —
the pattern that makes the graph batch-intolerant.

Run inside the cpu container (has onnx installed):
    docker exec glmocr-cpu python /work/scripts/analyze_layout_onnx.py
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import onnx
from onnx import numpy_helper, shape_inference

MODEL = Path("/root/.cache/huggingface/glmocr-layout-onnx/pp_doclayout_v3.onnx")


def load_with_shapes(path: Path) -> onnx.ModelProto:
    model = onnx.load(str(path))
    try:
        return shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"[warn] shape_inference failed: {e} — proceeding without")
        return model


def resolve_constant(name: str, initializers: dict, const_nodes: dict) -> list[int] | None:
    """Return the int values of a constant tensor named `name`, or None if it
    is not a pure constant we can follow."""
    if name in initializers:
        arr = numpy_helper.to_array(initializers[name])
        return arr.astype(int).tolist()
    if name in const_nodes:
        node = const_nodes[name]
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    arr = numpy_helper.to_array(attr.t)
                    return arr.astype(int).tolist()
    return None


def main() -> None:
    if not MODEL.exists():
        sys.exit(f"{MODEL} not found")
    print(f"[analyze] loading {MODEL} ({MODEL.stat().st_size / 1e6:.1f} MB)")
    model = load_with_shapes(MODEL)
    g = model.graph

    initializers = {init.name: init for init in g.initializer}
    const_nodes = {out: node for node in g.node if node.op_type == "Constant" for out in node.output}

    print(f"[analyze] {len(g.node)} nodes, {len(g.initializer)} initializers, opset {model.opset_import[0].version}")

    suspect_ops = ("Reshape", "Expand", "Tile")
    by_op = defaultdict(list)
    for node in g.node:
        if node.op_type in suspect_ops:
            by_op[node.op_type].append(node)

    for op, nodes in by_op.items():
        print(f"[analyze] {op}: {len(nodes)} nodes")

    suspects: list[tuple[str, str, list[int], str]] = []
    for op, nodes in by_op.items():
        shape_input_idx = 1  # all three ops take shape as input 1
        for node in nodes:
            if len(node.input) <= shape_input_idx:
                continue
            shape_name = node.input[shape_input_idx]
            values = resolve_constant(shape_name, initializers, const_nodes)
            if values is None:
                continue
            # batch-suspect: position 0 is literal 1, and value list has >=2 entries
            if len(values) >= 2 and values[0] == 1:
                data_in = node.input[0]
                suspects.append((node.name or "<anon>", op, values, data_in))

    print(f"\n[analyze] batch-1 suspects: {len(suspects)}")
    for name, op, shape, data_in in suspects:
        print(f"  {op:8s}  {name:40s}  shape={shape}  data_in={data_in}")

    # confirm the seed from the traceback
    seed = "node_view_320"
    hit = [s for s in suspects if s[0] == seed]
    print(f"\n[analyze] seed '{seed}' in suspects: {'YES' if hit else 'no (check other nodes)'}")


if __name__ == "__main__":
    main()
