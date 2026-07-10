"""Streamlit demo over the serving API: pick a user, watch the cascade produce their feed.

Two modes: replay a logged dataset user, or build a synthetic session ("your own taste") and
watch the same cascade react — the online path is user-id-free, so both hit identical models.
Deliberately self-contained — it talks to FastAPI over HTTP only (no vlmrec imports, no model
code), so it runs anywhere the API is reachable:  ``make demo``  or
``uv run streamlit run src/vlmrec/serving/demo_app.py`` (API base via $VLMREC_API).
"""

from __future__ import annotations

import json
import os
import random

import altair as alt
import pandas as pd
import requests
import streamlit as st

API = os.environ.get("VLMREC_API", "http://localhost:8000").rstrip("/")
ACCENT = "#4C78A8"  # single accent; neutrals carry everything else (color = identity, not decor)
NEUTRAL = "#9AA0A6"
STAGES = ["retrieval", "prerank", "rank", "final"]
STAGE_LABEL = {
    "retrieval": "candidate generation",
    "prerank": "pre-ranking",
    "rank": "ranking",
    "final": "post-processing",
}


# --- API helpers -----------------------------------------------------------------------------


@st.cache_data(ttl=600, show_spinner=False)
def api_get(path: str, **params):
    try:
        r = requests.get(f"{API}{path}", params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def api_post(path: str, payload: str):
    """POST with the JSON body passed as a string so st.cache_data can key on it."""
    try:
        r = requests.post(
            f"{API}{path}",
            data=payload,
            headers={"Content-Type": "application/json"},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def image_bytes(rel: str | None) -> bytes | None:
    """Fetch product photos through the UI process so the browser never needs the API origin."""
    if not rel:
        return None
    try:
        r = requests.get(f"{API}/images/{rel}", timeout=10)
        return r.content if r.ok else None
    except requests.RequestException:
        return None


def stage_survival(stages: dict, item_idx: int) -> tuple[str | None, int | None]:
    """Deepest cascade stage an item reached and its 1-based rank there."""
    reached, rank = None, None
    for name in STAGES:
        items = stages.get(name, {}).get("items", [])
        if item_idx in items:
            reached, rank = name, items.index(item_idx) + 1
        elif name in stages:
            break
    return reached, rank


# --- UI pieces -------------------------------------------------------------------------------


def item_card(col, it: dict, lines: list[str]):
    with col:
        img = image_bytes(it.get("image"))
        if img:
            st.image(img, use_container_width=True)
        else:
            st.markdown(
                "<div style='height:120px;display:flex;align-items:center;"
                "justify-content:center;font-size:2.5em'>🕹️</div>",
                unsafe_allow_html=True,
            )
        title = it["title"] if len(it["title"]) <= 60 else it["title"][:59] + "…"
        st.markdown(f"**{title}**")
        for line in lines:
            st.caption(line)


def meta_line(it: dict) -> str:
    bits = []
    if it.get("rating"):
        bits.append(f"★ {it['rating']:.1f} ({it['rating_n']})")
    if it.get("price") is not None:
        bits.append(f"${it['price']:.2f}")
    return " · ".join(bits) or " "


def card_grid(items: list[dict], lines_fn, per_row: int = 6):
    for start in range(0, len(items), per_row):
        row = items[start : start + per_row]
        for col, it in zip(st.columns(per_row), row, strict=False):  # last row may be short
            item_card(col, it, lines_fn(it))


def latency_chart(lat: dict):
    rows = [
        {"stage": STAGE_LABEL[s], "ms": lat[f"{k}_ms"]}
        for s, k in zip(STAGES, ["retrieve", "prerank", "rank", "postprocess"], strict=True)
    ]
    df = pd.DataFrame(rows)
    order = [STAGE_LABEL[s] for s in STAGES]
    base = alt.Chart(df).encode(
        y=alt.Y("stage:N", sort=order, title=None),
        x=alt.X("ms:Q", title="latency (ms)"),
        tooltip=[alt.Tooltip("stage:N"), alt.Tooltip("ms:Q", format=".2f")],
    )
    bars = base.mark_bar(color=ACCENT, cornerRadiusEnd=4, size=16)
    text = base.mark_text(align="left", dx=4).encode(text=alt.Text("ms:Q", format=".1f"))
    st.altair_chart((bars + text).properties(height=140), use_container_width=True)


def journey_chart(journey: list[dict], titles: dict[int, str], hit_idx: int | None):
    rows = []
    for j in journey:
        for s in STAGES:
            if j.get(s) is not None:
                rows.append(
                    {
                        "item": int(j["item_idx"]),
                        "title": titles.get(int(j["item_idx"]), str(j["item_idx"])),
                        "stage": STAGE_LABEL[s],
                        "rank": j[s],
                        "hit": int(j["item_idx"]) == hit_idx,
                    }
                )
    df = pd.DataFrame(rows)
    order = [STAGE_LABEL[s] for s in STAGES]
    chart = (
        alt.Chart(df)
        .mark_line(point=True, opacity=0.65)
        .encode(
            x=alt.X("stage:N", sort=order, title=None),
            y=alt.Y(
                "rank:Q",
                scale=alt.Scale(type="log", reverse=True),
                title="rank at stage (log, 1 = top)",
            ),
            detail="item:N",
            color=alt.condition(alt.datum.hit, alt.value(ACCENT), alt.value(NEUTRAL)),
            tooltip=["title:N", "stage:N", "rank:Q"],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)


def profile_panel(detail: dict):
    prof = detail.get("profile") or {}
    if not prof:
        st.caption("no VLM profile for this item")
        return
    if prof.get("one_line_summary"):
        st.markdown(f"> {prof['one_line_summary']}")
    chips = [
        prof.get("category_refined"),
        prof.get("sub_genre"),
        prof.get("target_audience"),
        prof.get("tone"),
    ]
    st.markdown(" · ".join(f"`{c}`" for c in chips if c))
    for label, key in [("Key attributes", "key_attributes"), ("Visual style", "visual_style")]:
        vals = prof.get(key) or []
        if isinstance(vals, list) and vals:
            st.markdown(f"**{label}:** " + " ".join(f"`{v}`" for v in vals[:8]))
    if prof.get("quality_cues"):
        st.caption(f"Quality cues: {prof['quality_cues']}")


# --- sections (shared by both modes) ---------------------------------------------------------


def render_cascade(rec: dict, hit_idx: int | None):
    stages, lat = rec["stages"], rec["latency_ms"]
    st.subheader("2 · The cascade run")
    cols = st.columns(5)
    counts = [len(stages[s]["items"]) if s in stages else None for s in STAGES]
    key = {"retrieval": "retrieve", "prerank": "prerank", "rank": "rank", "final": "postprocess"}
    for col, s, c in zip(cols, STAGES, counts, strict=False):  # 5th column = the total
        col.metric(STAGE_LABEL[s], c if c is not None else "skipped", f"{lat[key[s] + '_ms']} ms")
    cols[4].metric("end-to-end", f"{lat['total_ms']} ms", "CPU, single request")
    latency_chart(lat)
    if hit_idx is not None:
        reached, rank = stage_survival(stages, hit_idx)
        if reached == "final":
            st.success(f"✅ The held-out next purchase is in the final list at **#{rank}**.")
        elif reached is not None:
            st.info(
                f"The held-out item survived to **{STAGE_LABEL[reached]}** (rank {rank}) "
                "but was cut later — exactly the cascade-consistency problem WEEK7 studies."
            )
        else:
            st.caption(
                "The held-out item wasn't in the top-200 retrieval candidates this time "
                "(R@100 ≈ 0.21 — most single next-purchases aren't)."
            )


def render_feed(rec: dict, k: int, diversity: str):
    st.subheader(f"3 · Top-{k} recommendations ({diversity} diversity)")
    journey_by_item = {j["item_idx"]: j for j in rec["journey"]}

    def rec_lines(it: dict) -> list[str]:
        j = journey_by_item.get(it["item_idx"], {})
        move = f"cg #{j.get('retrieval', '–')} → final #{j.get('final', '–')}"
        return [meta_line(it), f"score {it['score']:.3f} · {move}"]

    card_grid(rec["results"], rec_lines, per_row=6)


def render_journey(rec: dict, titles: dict[int, str], hit_idx: int | None):
    st.subheader("4 · How the stages re-ordered things")
    st.caption(
        "Each line is one final item: its rank under candidate generation, after pre-rank, "
        "after the heavy ranker, and after diversity. Crossings = the ranker disagreeing "
        "with retrieval." + (" Blue = the held-out item." if hit_idx is not None else "")
    )
    journey_chart(rec["journey"], titles, hit_idx)


def render_inspector(rec: dict, titles: dict[int, str]):
    st.subheader("5 · Item inspector — what the VLM sees")
    pick = st.selectbox(
        "Inspect a recommended item",
        [r["item_idx"] for r in rec["results"]],
        format_func=lambda i: titles.get(i, str(i)),
    )
    detail = api_get(f"/item/{pick}")
    sim = api_get(f"/similar/{pick}", k=6)
    if detail:
        li, ri = st.columns([1, 3])
        item_card(li, detail, [meta_line(detail), detail.get("store", "")])
        with ri:
            st.markdown(
                "**Qwen2.5-VL structured profile** — generated from the product "
                "image + title, embedded as the third feature block"
            )
            profile_panel(detail)
    if sim:
        st.markdown("**Customers who liked this may also like** (item-tower neighbours)")
        card_grid(sim["similar"], lambda it: [meta_line(it), f"sim {it['score']:.3f}"], per_row=6)


def render_explainer():
    with st.expander("What am I looking at?"):
        st.markdown(
            """
- **Candidate generation** — a two-tower model (text + image + VLM-profile features) pulls
  ~200 candidates from a FAISS index over the full catalog, skipping items already interacted.
- **Pre-ranking** — a distilled lightweight ranker cuts 200 → 50 cheaply.
- **Ranking** — the heavy DIN + DCN-v2 + MMoE model scores click & satisfaction per item,
  using the behaviour sequence and the retrieval score as a cross-stage feature.
- **Post-processing** — scores are fused and MMR/DPP re-ranks for diversity.
- **Replay mode**: the held-out item is the user's *real* next purchase (temporal
  leave-last-out split), so the banner is an honest per-user glimpse of offline recall.
- **Build-your-own mode**: the user embedding is pooled item content and the ranker reads the
  behaviour *sequence*, never a user id — so a brand-new session works with zero retraining.
  That's also the cold-start story: content pathways generalize where ID embeddings can't.
            """
        )


# --- mode-specific section 1 -----------------------------------------------------------------


def replay_controls(sample: list[dict], n_users: int) -> int:
    if st.button("🎲 Surprise me", use_container_width=True) and sample:
        st.session_state["uid"] = random.choice(sample)["user_idx"]
    options = [u["user_idx"] for u in sample]
    hist = {u["user_idx"]: u["n_history"] for u in sample}
    default = st.session_state.get("uid", options[0] if options else 0)
    if default not in options:
        options = [default, *options]
    uid = st.selectbox(
        "User",
        options,
        index=options.index(default),
        format_func=lambda u: f"user {u} · {hist.get(u, '?')} items",
    )
    return int(st.number_input("…or any user_idx", 0, n_users - 1, uid))


def render_replay_user(prof: dict) -> int | None:
    st.subheader("1 · Who we're recommending for")
    test_it = prof.get("test_item")
    left, right = st.columns([4, 1])
    with left:
        st.markdown(f"Recent history — **{prof['n_history']} interactions**, newest first")
        card_grid(prof["history"], lambda it: [meta_line(it)], per_row=6)
    with right:
        st.markdown("**Held-out next purchase**")
        if test_it:
            item_card(st.container(), test_it, [meta_line(test_it)])
        else:
            st.caption("none for this user")
    return test_it["item_idx"] if test_it else None


def render_session_builder() -> list[int]:
    st.subheader("1 · Build your taste")
    basket: list[int] = st.session_state.setdefault("basket", [])
    q = st.text_input(
        "Search the catalog and add items you'd click on",
        placeholder='try "zelda", "gaming headset", "racing wheel", "ssd" …',
    )
    hits = api_get("/search", q=q.strip(), k=6) if len(q.strip()) >= 2 else None
    if hits:
        for col, it in zip(st.columns(6), hits, strict=False):
            item_card(col, it, [meta_line(it)])
            if col.button("➕ add", key=f"add_{it['item_idx']}", use_container_width=True):
                basket.append(it["item_idx"])
                st.rerun()
    elif hits == []:
        st.caption("no title matches — try fewer / different words")

    if basket:
        st.markdown(f"**Your session ({len(basket)} items, oldest first):**")
        brief = api_get("/items/brief", ids=",".join(map(str, basket))) or []
        for pos, (col, it) in enumerate(zip(st.columns(max(len(brief), 6)), brief, strict=False)):
            item_card(col, it, [meta_line(it)])
            if col.button("✖ remove", key=f"rm_{pos}", use_container_width=True):
                basket.pop(pos)
                st.rerun()
    else:
        st.info(
            "Add a few items you like — the cascade builds a feed for this brand-new "
            "'user' live. No user id, no retraining: pooled content + your sequence."
        )
    return basket


# --- page ------------------------------------------------------------------------------------


def main():
    st.set_page_config(page_title="VLM-Rec demo", page_icon="🎮", layout="wide")
    st.title("🎮 VLM-Rec — multimodal cascade, live")

    health = api_get("/health")
    if health is None:
        st.error(f"Serving API not reachable at `{API}` — start it with `make serve`.")
        st.stop()
    st.caption(
        f"{health['n_items']:,} items · {health['n_users']:,} users · "
        "candidate generation → pre-rank → rank → diversity, every request scored live"
    )

    sample = api_get("/users/sample", n=60, min_history=8, seed=7) or []
    with st.sidebar:
        st.header("Mode")
        mode = st.radio(
            "mode",
            ["Replay a real user", "Build your own taste"],
            label_visibility="collapsed",
        )
        st.header("Request")
        k = st.slider("Final list size (k)", 4, 30, 12)
        diversity = st.radio("Diversity re-rank", ["mmr", "dpp", "none"], horizontal=True)
        st.caption(
            "MMR/DPP trade a little relevance for variety in the final list — "
            "flip it and watch near-duplicates drop out."
        )
        uid = None
        if mode == "Replay a real user":
            st.header("User")
            uid = replay_controls(sample, health["n_users"])

    if mode == "Replay a real user":
        rec = api_get(
            "/recommend", user_id=uid, k=k, diversity=diversity, explain=True, enrich=True
        )
        prof = api_get(f"/user/{uid}", n=6)
        if rec is None or prof is None:
            st.error("recommend call failed — check the API logs")
            st.stop()
        hit_idx = render_replay_user(prof)
    else:
        basket = render_session_builder()
        if not basket:
            render_explainer()
            st.stop()
        payload = json.dumps(
            {"items": basket, "k": k, "diversity": diversity, "explain": True, "enrich": True},
            sort_keys=True,
        )
        rec = api_post("/recommend/session", payload)
        if rec is None:
            st.error("session recommend call failed — check the API logs")
            st.stop()
        hit_idx = None

    render_cascade(rec, hit_idx)
    render_feed(rec, k, diversity)
    titles = {r["item_idx"]: r["title"] for r in rec["results"]}
    render_journey(rec, titles, hit_idx)
    render_inspector(rec, titles)
    render_explainer()


if __name__ == "__main__":  # streamlit executes the script as __main__
    main()
