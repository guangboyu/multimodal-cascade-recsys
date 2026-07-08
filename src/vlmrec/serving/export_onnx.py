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


class _ExportRankerScore(nn.Module):
    """Variant for rankers with the cross-stage retrieval-score input."""

    def __init__(self, ranker, content_pad, cat):
        super().__init__()
        self.ranker = ranker
        self.register_buffer("content_pad", content_pad)
        self.register_buffer("cat", cat)

    def forward(self, cand, seq, ret_score):
        return self.ranker(cand, seq, self.content_pad, self.cat, ret_score=ret_score)


def export(cfg=None) -> dict:
    cfg = cfg or load_config()
    paths = Paths(cfg)
    reg = load_registry(cfg)
    use_score = bool(getattr(reg.ranker, "use_retrieval_score", False))

    n = 16
    cand = torch.randint(0, reg.d.n_items, (n,))
    seq = reg.seq_t[torch.zeros(n, dtype=torch.long)]
    if use_score:
        wrapper = _ExportRankerScore(reg.ranker, reg.content, reg.cat).eval()
        args = (cand, seq, torch.rand(n) * 2 - 1)  # retrieval scores live in [-1, 1]
        input_names = ["cand", "seq", "ret_score"]
    else:
        wrapper = _ExportRanker(reg.ranker, reg.content, reg.cat).eval()
        args = (cand, seq)
        input_names = ["cand", "seq"]

    out_dir = paths.data / "serving"
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "ranker.onnx"
    torch.onnx.export(
        wrapper,
        args,
        str(onnx_path),
        input_names=input_names,
        output_names=["logits"],
        dynamic_axes={name: {0: "n"} for name in [*input_names, "logits"]},
        opset_version=17,
        dynamo=False,
    )

    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        torch_out = wrapper(*args).numpy()
    feeds = {name: t.numpy() for name, t in zip(input_names, args, strict=True)}
    onnx_out = sess.run(None, feeds)[0]
    max_diff = float(np.abs(torch_out - onnx_out).max())
    log.info("ONNX export -> %s | parity max|diff|=%.2e", onnx_path, max_diff)
    return {"path": str(onnx_path), "max_abs_diff": max_diff, "inputs": input_names}


def run(cfg, paths: Paths) -> dict:
    return export(cfg)
