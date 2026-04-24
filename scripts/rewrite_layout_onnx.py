"""Rewrite PP-DocLayoutV3 ONNX to be batch-dim-invariant.

For every Reshape whose constant shape tensor starts with literal 1, swap
position 0 to 0 ("copy from input" per Reshape spec, opset >= 14). This
lets batch>1 flow through without forcing a second -1 (illegal).

Expand/Tile nodes left untouched — their semantics differ and the
failing-batch error pointed only at Reshape (node_view_320).

Outputs: pp_doclayout_v3_dyn.onnx + .data alongside the original.
Run:
    docker exec -i glmocr-cpu python - < scripts/rewrite_layout_onnx.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

SRC = Path("/root/.cache/huggingface/glmocr-layout-onnx/pp_doclayout_v3.onnx")
DST = SRC.with_name("pp_doclayout_v3_dyn.onnx")


def main() -> None:
    if not SRC.exists():
        sys.exit(f"{SRC} not found")
    print(f"[rewrite] loading {SRC}")
    model = onnx.load(str(SRC), load_external_data=True)
    g = model.graph

    initializers = {init.name: init for init in g.initializer}
    const_nodes = {out: node for node in g.node for out in node.output if node.op_type == "Constant"}

    # Locate Reshapes whose shape initializer starts with literal 1.
    targets: list[str] = []
    for node in g.node:
        if node.op_type != "Reshape" or len(node.input) < 2:
            continue
        shape_name = node.input[1]

        # initializer case — mutate in place
        if shape_name in initializers:
            init = initializers[shape_name]
            arr = numpy_helper.to_array(init)
            if arr.ndim == 1 and len(arr) >= 2 and int(arr[0]) == 1:
                new_arr = arr.copy()
                new_arr[0] = 0  # "copy from input dim"
                # rebuild initializer
                new_init = numpy_helper.from_array(new_arr.astype(arr.dtype), name=init.name)
                init.CopyFrom(new_init)
                targets.append(node.name or shape_name)
            continue

        # Constant-node case
        if shape_name in const_nodes:
            c = const_nodes[shape_name]
            for attr in c.attribute:
                if attr.name == "value":
                    arr = numpy_helper.to_array(attr.t)
                    if arr.ndim == 1 and len(arr) >= 2 and int(arr[0]) == 1:
                        new_arr = arr.copy()
                        new_arr[0] = 0
                        new_t = numpy_helper.from_array(new_arr.astype(arr.dtype), name=attr.t.name)
                        attr.t.CopyFrom(new_t)
                        targets.append(node.name or shape_name)

    print(f"[rewrite] patched {len(targets)} Reshape nodes; sample: {targets[:5]}...")

    # Belt-and-braces: ensure every Reshape op has allowzero unset / 0 so
    # our zero is interpreted as "copy from input" rather than literal zero.
    for node in g.node:
        if node.op_type != "Reshape":
            continue
        for attr in node.attribute:
            if attr.name == "allowzero" and attr.i == 1:
                attr.i = 0
                print(f"[rewrite] reset allowzero=0 on {node.name}")

    # Save, preserving external data.
    data_name = DST.name + ".data"
    onnx.save(
        model,
        str(DST),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_name,
    )
    print(f"[rewrite] wrote {DST} ({DST.stat().st_size / 1e6:.1f} MB) + {data_name}")


if __name__ == "__main__":
    main()
