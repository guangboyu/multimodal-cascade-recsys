"""Export the heavy ranker to ONNX (item/feature tables baked in as buffers) + verify parity.

Demonstrates the offline→serving model-export path. The exported graph takes ``(cand, seq)`` with a
dynamic batch axis; the large content/cat tables are frozen as constants so the artifact is
self-contained for an ONNX Runtime / Triton deployment.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..config import load_config
from ..paths import Paths
from ..utils import get_logger
from .registry import load_registry

log = get_logger("vlmrec.serving.onnx")


class _ExportRanker(nn.Module):
    def __init__(self, ranker, content_pad, cat):
        super().__init__()
        self.ranker = ranker
        self.register_buffer("content_pad", content_pad)
        self.register_buffer("cat", cat)

    def forward(self, cand, seq):
        return self.ranker(cand, seq, self.content_pad, self.cat)


def export(cfg=None) -> dict:
    cfg = cfg or load_config()
    paths = Paths(cfg)
    reg = load_registry(cfg)
    wrapper = _ExportRanker(reg.ranker, reg.content, reg.cat).eval()

    n = 16
    cand = torch.randint(0, reg.d.n_items, (n,))
    seq = reg.seq_t[torch.zeros(n, dtype=torch.long)]

    out_dir = paths.data / "serving"
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "ranker.onnx"
    torch.onnx.export(
        wrapper,
        (cand, seq),
        str(onnx_path),
        input_names=["cand", "seq"],
        output_names=["logits"],
        dynamic_axes={"cand": {0: "n"}, "seq": {0: "n"}, "logits": {0: "n"}},
        opset_version=17,
        dynamo=False,
    )

    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        torch_out = wrapper(cand, seq).numpy()
    onnx_out = sess.run(None, {"cand": cand.numpy(), "seq": seq.numpy()})[0]
    max_diff = float(np.abs(torch_out - onnx_out).max())
    log.info("ONNX export -> %s | parity max|diff|=%.2e", onnx_path, max_diff)
    return {"path": str(onnx_path), "max_abs_diff": max_diff}


def run(cfg, paths: Paths) -> dict:
    return export(cfg)
