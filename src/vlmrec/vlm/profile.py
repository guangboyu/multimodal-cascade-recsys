"""VLM item understanding: batch-generate structured JSON item profiles (Qwen2.5-VL).

Each item's product image + title/description goes through a vision-language model that returns
a fixed-schema JSON profile (refined category, visual style, attributes, audience, tone).
Profiles are stored as JSON strings for a stable parquet schema (the Week-1 ingestion pattern),
then embedded by ``encode_profile``. Inference is shard-checkpointed and resumable — this is a
multi-hour batch job, and re-running skips completed shards.
"""

from __future__ import annotations

import json
import time

import polars as pl

from ..features.encode_text import _field_text
from ..paths import Paths
from ..utils import get_logger

log = get_logger("vlmrec.vlm.profile")

SYSTEM_PROMPT = (
    "You are a product-catalog analyst. Given a product image and its store listing text, reply "
    "with ONLY a JSON object — no code fences, no commentary — with exactly these keys: "
    "category_refined (string), sub_genre (string), visual_style (array of 2-4 short strings), "
    "key_attributes (array of 3-6 short strings), target_audience (string), tone (string), "
    "quality_cues (string), one_line_summary (string, at most 25 words)."
)

PROFILE_SCHEMA = {
    "type": "object",
    "properties": {
        "category_refined": {"type": "string"},
        "sub_genre": {"type": "string"},
        "visual_style": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
        "key_attributes": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
        "target_audience": {"type": "string"},
        "tone": {"type": "string"},
        "quality_cues": {"type": "string"},
        "one_line_summary": {"type": "string"},
    },
    "required": [
        "category_refined",
        "sub_genre",
        "visual_style",
        "key_attributes",
        "target_audience",
        "tone",
        "quality_cues",
        "one_line_summary",
    ],
}


def default_profile(title: str = "") -> dict:
    return {
        "category_refined": "",
        "sub_genre": "",
        "visual_style": [],
        "key_attributes": [],
        "target_audience": "",
        "tone": "",
        "quality_cues": "",
        "one_line_summary": (title or "")[:120],
    }


def parse_profile(raw: str, title: str = "") -> tuple[dict, int]:
    """Defensive parse of model output -> (profile with every schema key, ok flag).

    Tolerates code fences / leading prose by slicing the outermost {...}; any failure returns
    a default profile (one_line_summary = the item title) with ok=0 so shapes stay intact.
    """
    s = raw or ""
    a, b = s.find("{"), s.rfind("}")
    if a < 0 or b <= a:
        return default_profile(title), 0
    try:
        obj = json.loads(s[a : b + 1])
    except json.JSONDecodeError:
        return default_profile(title), 0
    if not isinstance(obj, dict):
        return default_profile(title), 0
    out = default_profile(title)
    found = 0
    for k, dv in out.items():
        v = obj.get(k)
        if v is None:
            continue
        found += 1
        if isinstance(dv, list):
            out[k] = [str(x) for x in v][:8] if isinstance(v, list) else [str(v)]
        elif isinstance(v, str):
            out[k] = v
        elif isinstance(v, list):  # list where a string was expected: join, don't repr()
            out[k] = ", ".join(str(x) for x in v)
        else:
            out[k] = str(v)
    # a parseable dict with (almost) none of the schema keys is a failure, not a success
    return out, (1 if found >= 3 else 0)


def build_prompt_rows(paths: Paths, max_desc_chars: int = 800) -> list[dict]:
    """One prompt row per item (ordered by item_idx): title/description text + image path."""
    item_map = pl.read_parquet(paths.item_map_parquet)
    meta = pl.read_parquet(paths.meta_parquet)
    merged = item_map.join(meta, on="parent_asin", how="left").sort("item_idx")
    rows = []
    for r in merged.iter_rows(named=True):
        title = _field_text(r.get("title"))
        desc = _field_text(r.get("description"))[:max_desc_chars]
        img = paths.image_file(r["parent_asin"])
        rows.append(
            {
                "item_idx": int(r["item_idx"]),
                "title": title,
                "text": f"Title: {title}\nDescription: {desc}",
                "image_path": str(img) if img.exists() else None,
            }
        )
    return rows


def _user_content(row: dict) -> list[dict]:
    parts = []
    if row["image_path"]:
        parts.append({"type": "image_url", "image_url": {"url": f"file://{row['image_path']}"}})
    parts.append({"type": "text", "text": row["text"]})
    return parts


class _VllmBackend:
    name = "vllm"

    def __init__(self, v):
        from vllm import LLM, SamplingParams

        kwargs = dict(
            model=str(v.model),
            max_model_len=2048,
            gpu_memory_utilization=float(v.gpu_memory_utilization),
            limit_mm_per_prompt={"image": 1},
            mm_processor_kwargs={"max_pixels": int(v.max_pixels)},
        )
        self.llm = LLM(**kwargs)
        guided = None
        try:
            from vllm.sampling_params import GuidedDecodingParams

            guided = GuidedDecodingParams(json=PROFILE_SCHEMA)
        except ImportError:
            log.warning("vLLM guided decoding unavailable — relying on defensive parsing")
        self.sp = SamplingParams(
            temperature=float(v.temperature),
            max_tokens=int(v.max_new_tokens),
            guided_decoding=guided,
        )

    def generate(self, rows: list[dict]) -> list[str]:
        msgs = [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _user_content(r)},
            ]
            for r in rows
        ]
        outs = self.llm.chat(msgs, self.sp, use_tqdm=False)
        return [o.outputs[0].text for o in outs]


class _TransformersBackend:
    name = "transformers"

    def __init__(self, v, model_id: str | None = None, batch_size: int = 8):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        model_id = model_id or str(v.model)
        self.name = f"transformers:{model_id}"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="cuda"
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_id, max_pixels=int(v.max_pixels))
        # decoder-only generation must left-pad: right-padding makes every non-longest row in
        # the batch generate from pad tokens -> garbage (classic batched-generate bug)
        self.processor.tokenizer.padding_side = "left"
        self.max_new_tokens = int(v.max_new_tokens)
        self.batch_size = batch_size

    def _chat(self, row: dict) -> list[dict]:
        content = []
        if row["image_path"]:
            content.append({"type": "image", "image": f"file://{row['image_path']}"})
        content.append({"type": "text", "text": row["text"]})
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

    def generate(self, rows: list[dict]) -> list[str]:
        import torch
        from qwen_vl_utils import process_vision_info

        out_by_idx: dict[int, str] = {}
        # image-less rows go through separate batches: the processor can't mix modalities
        groups = [
            [i for i, r in enumerate(rows) if r["image_path"]],
            [i for i, r in enumerate(rows) if not r["image_path"]],
        ]
        for group in groups:
            for s in range(0, len(group), self.batch_size):
                idxs = group[s : s + self.batch_size]
                msgs = [self._chat(rows[i]) for i in idxs]
                texts = [
                    self.processor.apply_chat_template(
                        m, tokenize=False, add_generation_prompt=True
                    )
                    for m in msgs
                ]
                images, _ = process_vision_info(msgs)
                inputs = self.processor(
                    text=texts, images=images or None, padding=True, return_tensors="pt"
                ).to(self.model.device)
                with torch.no_grad():
                    gen = self.model.generate(
                        **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
                    )
                trimmed = [g[len(i) :] for i, g in zip(inputs.input_ids, gen, strict=True)]
                decoded = self.processor.batch_decode(trimmed, skip_special_tokens=True)
                for i, d in zip(idxs, decoded, strict=True):
                    out_by_idx[i] = d
        return [out_by_idx[i] for i in range(len(rows))]


def _make_backend(cfg):
    v = cfg.vlm
    if str(v.backend) == "vllm":
        try:
            return _VllmBackend(v)
        except ImportError as e:
            # e.g. the PyPI vllm wheel links a newer libcudart than the driver supports —
            # the cu128 pitfall repeating at the inference-engine layer (docs/PITFALLS.md)
            log.warning("vLLM unavailable (%s) — falling back to transformers backend", e)
            return _TransformersBackend(v, model_id=str(v.get("fallback_model", v.model)))
    return _TransformersBackend(v)


def run(cfg, paths: Paths) -> dict:
    v = cfg.vlm
    shard_dir = paths.vlm / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    rows = build_prompt_rows(paths, int(v.max_desc_chars))
    shard = int(v.shard_size)
    n_shards = (len(rows) + shard - 1) // shard
    # resume safety: shards are keyed by index, so the chunking must match the previous run
    manifest_path = shard_dir / "manifest.json"
    if manifest_path.exists():
        prev = json.loads(manifest_path.read_text())
        if prev.get("shard_size") != shard or prev.get("n_items") != len(rows):
            raise RuntimeError(
                f"shard layout changed (was {prev}, now shard_size={shard} n_items={len(rows)})"
                f" — delete {shard_dir} to restart cleanly instead of mixing chunkings"
            )
    else:
        manifest_path.write_text(json.dumps({"shard_size": shard, "n_items": len(rows)}))
    log.info("profiling %s items in %d shards of %d", f"{len(rows):,}", n_shards, shard)

    backend = None
    t0 = time.time()
    for si in range(n_shards):
        out = shard_dir / f"shard_{si:05d}.parquet"
        if out.exists():
            continue
        if backend is None:  # lazy: resume-only runs never load the model
            backend = _make_backend(cfg)
        chunk = rows[si * shard : (si + 1) * shard]
        raws = backend.generate(chunk)
        recs = []
        for r, raw in zip(chunk, raws, strict=True):
            prof, ok = parse_profile(raw, r["title"])
            recs.append(
                {
                    "item_idx": r["item_idx"],
                    "profile_json": json.dumps(prof, ensure_ascii=False),
                    "ok": ok,
                }
            )
        tmp = out.with_suffix(".tmp.parquet")  # atomic publish: a crash never fakes a done shard
        pl.DataFrame(recs).write_parquet(tmp)
        tmp.rename(out)
        log.info("shard %d/%d done | %.1f min elapsed", si + 1, n_shards, (time.time() - t0) / 60)

    df = pl.concat(
        [pl.read_parquet(shard_dir / f"shard_{si:05d}.parquet") for si in range(n_shards)]
    ).sort("item_idx")
    assert df.height == len(rows), f"profile count mismatch: {df.height} vs {len(rows)}"
    df.write_parquet(paths.profiles_parquet)
    meta = {
        "model": str(v.model),
        "backend": backend.name if backend else "resumed (all shards cached)",
        "n_items": int(df.height),
        "validity_rate": round(float(df.get_column("ok").mean()), 4),
        "wall_clock_min": round((time.time() - t0) / 60, 1),
    }
    (paths.vlm / "profile_meta.json").write_text(json.dumps(meta, indent=2))
    log.info("profiles -> %s | %s", paths.profiles_parquet, meta)
    return meta
