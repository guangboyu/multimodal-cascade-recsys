"""Pure-logic tests for the VLM profile pipeline (parsing, templates, source selection)."""

from __future__ import annotations

import json

import pytest

from vlmrec.retrieval.data import canonical_sources
from vlmrec.vlm.encode_profile import profile_to_text
from vlmrec.vlm.profile import default_profile, parse_profile

GOOD = {
    "category_refined": "Action RPG",
    "sub_genre": "soulslike",
    "visual_style": ["dark fantasy", "gothic"],
    "key_attributes": ["open world", "co-op", "60fps"],
    "target_audience": "hardcore gamers",
    "tone": "grim",
    "quality_cues": "premium collector packaging",
    "one_line_summary": "A punishing dark-fantasy action RPG.",
}


def test_parse_clean_json():
    prof, ok = parse_profile(json.dumps(GOOD))
    assert ok == 1
    assert prof == GOOD


def test_parse_fenced_json_and_prose_prefix():
    raw = "Sure! Here is the profile:\n```json\n" + json.dumps(GOOD) + "\n```"
    prof, ok = parse_profile(raw)
    assert ok == 1
    assert prof["category_refined"] == "Action RPG"


def test_parse_garbage_falls_back_to_title():
    prof, ok = parse_profile("not json at all", title="Halo 3")
    assert ok == 0
    assert prof == default_profile("Halo 3")
    assert prof["one_line_summary"] == "Halo 3"


def test_parse_coerces_wrong_types():
    obj = dict(GOOD, visual_style="minimalist", key_attributes=[1, 2])
    prof, ok = parse_profile(json.dumps(obj))
    assert ok == 1
    assert prof["visual_style"] == ["minimalist"]  # bare string -> singleton list
    assert prof["key_attributes"] == ["1", "2"]  # numbers -> strings


def test_parse_missing_keys_keep_defaults():
    # fewer than 3 schema keys present -> the profile is too empty to count as valid
    prof, ok = parse_profile(json.dumps({"category_refined": "Puzzle"}), title="Tetris")
    assert ok == 0
    assert prof["category_refined"] == "Puzzle"  # partial content is still kept
    assert prof["visual_style"] == []
    assert prof["one_line_summary"] == "Tetris"  # default carries the title


def test_parse_schema_free_dict_is_a_failure():
    assert parse_profile('{"unrelated": 1}')[1] == 0
    assert parse_profile("{}")[1] == 0


def test_parse_list_for_string_field_joins_cleanly():
    obj = dict(GOOD, target_audience=["teens", "adults"])
    prof, ok = parse_profile(json.dumps(obj))
    assert ok == 1
    assert prof["target_audience"] == "teens, adults"  # no Python repr() garbage


def test_profile_to_text_deterministic_and_flat():
    a = profile_to_text(GOOD)
    b = profile_to_text(json.loads(json.dumps(GOOD)))
    assert a == b
    assert "dark fantasy, gothic" in a
    assert "\n" not in a


def test_profile_to_text_empty_profile_falls_back_to_title():
    assert profile_to_text(default_profile(""), title="Mario Kart") == "Mario Kart"


def test_canonical_sources_orders_and_validates():
    assert canonical_sources(("image", "text")) == ("text", "image")
    assert canonical_sources(["vlm", "image"]) == ("image", "vlm")
    with pytest.raises(ValueError):
        canonical_sources(("audio",))
    with pytest.raises(ValueError):
        canonical_sources(())
