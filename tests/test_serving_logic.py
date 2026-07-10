"""Pure-logic tests for the serving/demo helpers: profile parsing, titles, stage journeys."""

from vlmrec.serving.catalog import display_title, parse_profile
from vlmrec.serving.service import stage_journey


def test_parse_profile_valid():
    assert parse_profile('{"sub_genre": "RPG"}') == {"sub_genre": "RPG"}


def test_parse_profile_garbage_and_empty():
    assert parse_profile("not json {") == {}
    assert parse_profile("") == {}
    assert parse_profile(None) == {}


def test_parse_profile_non_dict_json():
    # a bare list/string is valid JSON but not a profile — must not leak through
    assert parse_profile('["a", "b"]') == {}
    assert parse_profile('"just a string"') == {}


def test_display_title_falls_back_to_asin():
    assert display_title("", "B0TEST") == "[B0TEST]"
    assert display_title(None, "B0TEST") == "[B0TEST]"
    assert display_title("  ", "B0TEST") == "[B0TEST]"


def test_display_title_truncates():
    long = "x" * 200
    out = display_title(long, "B0TEST", max_chars=50)
    assert len(out) == 50 and out.endswith("…")
    assert display_title("short", "B0TEST") == "short"


def test_stage_journey_tracks_ranks_across_stages():
    stages = {
        "retrieval": {"items": [5, 3, 9, 7], "scores": [0.9, 0.8, 0.7, 0.6]},
        "prerank": {"items": [3, 9, 5], "scores": [0.9, 0.8, 0.7]},
        "rank": {"items": [9, 3, 5], "scores": [0.9, 0.8, 0.7]},
        "final": {"items": [9, 5], "scores": [0.9, 0.7]},
    }
    j = stage_journey(stages, [9, 5])
    assert j[0] == {"item_idx": 9, "retrieval": 3, "prerank": 2, "rank": 1, "final": 1}
    assert j[1] == {"item_idx": 5, "retrieval": 1, "prerank": 3, "rank": 3, "final": 2}


def test_stage_journey_absent_item_is_none():
    stages = {"retrieval": {"items": [1, 2]}, "final": {"items": [7]}}
    (j,) = stage_journey(stages, [7])
    assert j["retrieval"] is None and j["final"] == 1
