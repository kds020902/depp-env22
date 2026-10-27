"""Self-contained HTML report for predict.py --report: one file with the photos embedded, opens offline in any browser.

Summary counts, a per-vegetable table, filters (rotten / take again / turned away / fresh) and one card per photo with
its verdict, probability, rot-evidence meter, retake advice or closest-thing hint, and the rot heatmap on a button.
"""

from __future__ import annotations

import base64
import html
import io
from datetime import datetime
from pathlib import Path

from PIL import Image

SPECIES_ORDER = ["오이", "감자", "토마토", "피망", "당근"]


def image_data_uri(image: Image.Image, size: int = 360, quality: int = 80) -> str:
    image = image.convert("RGB").copy()
    image.thumbnail((size, size))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _pct(value: float) -> str:
    return "99% 이상" if value >= 0.995 else "1% 미만" if value < 0.005 else f"{round(value * 100)}%"


ATTENTION = {"rotten": 0, "unknown": 2, "error": 3, "fresh": 4}


def _card(entry: dict) -> str:
    e = {k: html.escape(str(v)) if isinstance(v, str) else v for k, v in entry.items()}
    verdict = entry["verdict"]
    retake = entry.get("retake") and verdict in ("fresh", "rotten")
    if verdict in ("fresh", "rotten"):
        head = (f'<div class="verdict"><span class="veg">{e["species"]}</span>'
                f'<span class="chip {verdict}">{e["condition"]}</span></div>'
                f'<p class="prob">이 판정의 확률 <b>{_pct(entry["probability"])}</b></p>')
        rot = entry.get("rot")
        meter = "" if rot is None else (
            f'<div class="rot" title="상한 흔적: Valid 비정상 사진의 중간 수준을 1로 본 값">'
            f'<span class="rot-label">상한 흔적</span><span class="meter"><span style="width:{min(1.0, rot) * 100:.0f}%"></span></span>'
            f'<span class="rot-val">{rot:.2f}</span></div>')
        body = head + meter
    elif verdict == "unknown":
        body = ('<div class="verdict"><span class="chip unknown">학습된 데이터가 아닙니다</span></div>'
                + (f'<p class="prob">가장 가까운 것: <b>{e["closest"]}</b></p>' if entry.get("closest") else ""))
    else:
        body = f'<div class="verdict"><span class="chip unknown">열 수 없음</span></div><p class="prob">{e.get("error", "")}</p>'
    notes = ""
    if retake:
        notes += f'<p class="note warn">{e["retake_message"]}</p>'
    if verdict == "unknown" and entry.get("hint"):
        notes += f'<p class="note">{e["hint"]}</p>'
    shot = ""
    if entry.get("photo_uri"):
        heat = (f'<img class="heat" src="{entry["heat_uri"]}" alt="{e["name"]}: 상한 곳 표시" hidden>'
                if entry.get("heat_uri") else "")
        shot = f'<figure class="shot"><img class="photo" src="{entry["photo_uri"]}" alt="{e["name"]}">{heat}</figure>'
    button = ('<button class="heat-btn" type="button" aria-pressed="false">상한 곳 보기</button>'
              if entry.get("heat_uri") else "")
    flags = f'data-verdict="{verdict}" data-retake="{1 if retake else 0}"'
    return (f'<article class="card" {flags}>{shot}<div class="card-body">'
            f'<p class="name" title="{e["name"]}">{e["name"]}</p>{body}{notes}{button}</div></article>')


def write_report(path: Path, entries: list[dict], meta: dict) -> None:
    """entries: dicts with name, verdict (fresh/rotten/unknown/error), species, condition (Korean), probability,
    retake, retake_message, closest, hint, rot, photo_uri, heat_uri, error. meta: model, source, notes."""
    counts = {v: sum(e["verdict"] == v for e in entries) for v in ("fresh", "rotten", "unknown", "error")}
    retakes = sum(bool(e.get("retake")) and e["verdict"] in ("fresh", "rotten") for e in entries)
    table = {s: [0, 0] for s in SPECIES_ORDER}
    for e in entries:
        if e["verdict"] in ("fresh", "rotten"):
            table[e["species"]][e["verdict"] == "rotten"] += 1
    rows = "".join(f'<tr><th scope="row">{s}</th><td>{a}</td><td>{b}</td><td>{a + b}</td></tr>'
                   for s, (a, b) in table.items() if a + b)
    ordered = sorted(entries, key=lambda e: (1 if e["verdict"] == "fresh" and e.get("retake") else ATTENTION[e["verdict"]],
                                             e["name"]))
    cards = "\n".join(_card(e) for e in ordered)
    tiles = [("전체", len(entries), "all"), ("비정상", counts["rotten"], "rotten"), ("다시 찍기 권장", retakes, "retake"),
             ("학습된 데이터가 아닙니다", counts["unknown"] + counts["error"], "unknown"), ("정상", counts["fresh"], "fresh")]
    tile_html = "".join(f'<button class="tile t-{k}" type="button" data-filter="{k}" aria-pressed="{str(k == "all").lower()}">'
                        f'<span class="tile-n">{n}</span><span class="tile-l">{label}</span></button>' for label, n, k in tiles)
    notes = "".join(f"<li>{html.escape(n)}</li>" for n in meta.get("notes", []))
    page = TEMPLATE.format(
        generated=datetime.now().strftime("%Y-%m-%d %H:%M"), model=html.escape(meta.get("model", "")),
        source=html.escape(meta.get("source", "")), total=len(entries), tiles=tile_html,
        table=rows or '<tr><td colspan="4">판정된 채소가 없습니다</td></tr>', cards=cards, notes=notes)
    Path(path).write_text(page, encoding="utf-8")


TEMPLATE = """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>채소 판별 리포트</title>
<style>
/* Layout: an inspection log. Counts first (they double as filters), a small tally by vegetable, then one card per photo,
   photos that need a look first. Works offline: system fonts only, every image embedded. */
:root {{
  --bg: #f3f5f0; --surface: #ffffff; --ink: #1b221d; --ink-2: #59625c; --line: #d8ddd3; --track: #e6e9e2;
  --accent: #2552b8; --fresh: #2e7a33; --fresh-bg: #e1eedf; --rotten: #a24d24; --rotten-bg: #f6e2d6;
  --warn: #8a5a00; --warn-bg: #fbf0d6; --muted-bg: #eceee9;
  --font-body: "Apple SD Gothic Neo", "Malgun Gothic", "Noto Sans KR", system-ui, sans-serif;
  --font-data: ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
  --radius: 10px;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg: #111512; --surface: #191f1a; --ink: #e7ebe5; --ink-2: #a2aca4; --line: #2b342d; --track: #27302a;
    --accent: #84a6ff; --fresh: #72c276; --fresh-bg: #1e2f20; --rotten: #e3926a; --rotten-bg: #3b2419;
    --warn: #f0c46a; --warn-bg: #3a2f17; --muted-bg: #232a24; color-scheme: dark;
  }}
}}
:root[data-theme="dark"] {{
  --bg: #111512; --surface: #191f1a; --ink: #e7ebe5; --ink-2: #a2aca4; --line: #2b342d; --track: #27302a;
  --accent: #84a6ff; --fresh: #72c276; --fresh-bg: #1e2f20; --rotten: #e3926a; --rotten-bg: #3b2419;
  --warn: #f0c46a; --warn-bg: #3a2f17; --muted-bg: #232a24; color-scheme: dark;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--bg); color: var(--ink); font-family: var(--font-body); font-size: 15px; line-height: 1.55; }}
.wrap {{ max-width: 1120px; margin: 0 auto; padding-block: 28px 48px; padding-left: max(16px, env(safe-area-inset-left, 0px));
  padding-right: max(16px, env(safe-area-inset-right, 0px)); display: grid; gap: 22px; }}
header {{ display: grid; gap: 6px; }}
h1 {{ margin: 0; font-size: 1.9rem; line-height: 1.2; text-wrap: balance; }}
h2 {{ margin: 0; font-size: 1.05rem; }}
.meta {{ display: flex; flex-wrap: wrap; gap: 4px 16px; color: var(--ink-2); font-size: 0.8125rem; font-family: var(--font-data); }}
.tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }}
.tile {{ font: inherit; text-align: left; cursor: pointer; display: grid; gap: 2px; padding: 12px 14px; border-radius: var(--radius);
  background: var(--surface); color: var(--ink); border: 1px solid var(--line); }}
.tile-n {{ font-family: var(--font-data); font-size: 1.6rem; font-weight: 600; font-variant-numeric: tabular-nums; }}
.tile-l {{ color: var(--ink-2); font-size: 0.8125rem; }}
.tile[aria-pressed="true"] {{ border-color: var(--accent); box-shadow: 0 0 0 2px var(--accent); }}
.tile.t-rotten .tile-n {{ color: var(--rotten); }} .tile.t-retake .tile-n {{ color: var(--warn); }} .tile.t-fresh .tile-n {{ color: var(--fresh); }}
.tile:focus-visible, .heat-btn:focus-visible, .switch input:focus-visible {{ outline: 3px solid var(--accent); outline-offset: 2px; }}
.tally {{ display: grid; gap: 8px; min-width: 0; }}
.table-scroll {{ overflow-x: auto; }}
table {{ border-collapse: collapse; background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); min-width: 320px; }}
th, td {{ padding: 6px 14px; border-bottom: 1px solid var(--line); text-align: right; font-variant-numeric: tabular-nums; }}
thead th {{ font-size: 0.75rem; color: var(--ink-2); font-weight: 600; letter-spacing: 0.04em; }}
th[scope="row"], thead th:first-child {{ text-align: left; }}
tr:last-child th, tr:last-child td {{ border-bottom: 0; }}
.toolbar {{ display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 10px; }}
.switch {{ display: inline-flex; gap: 8px; align-items: center; font-size: 0.875rem; color: var(--ink-2); cursor: pointer; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); gap: 12px; }}
.card {{ background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); overflow: hidden; display: grid;
  grid-template-rows: auto 1fr; min-width: 0; }}
.card[data-verdict="rotten"] {{ border-color: var(--rotten); }}
.shot {{ margin: 0; aspect-ratio: 4 / 3; background: var(--track); display: grid; place-items: center; overflow: hidden; max-width: 100%; }}
.shot img {{ width: 100%; height: 100%; object-fit: contain; display: block; }}
.card-body {{ padding: 10px 12px 12px; display: grid; gap: 6px; align-content: start; min-width: 0; }}
.name {{ margin: 0; font-family: var(--font-data); font-size: 0.75rem; color: var(--ink-2); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.verdict {{ display: flex; flex-wrap: wrap; gap: 6px 8px; align-items: baseline; }}
.veg {{ font-size: 1.25rem; font-weight: 700; }}
.chip {{ font-size: 0.8125rem; font-weight: 700; padding: 1px 10px; border-radius: 999px; border: 1.5px solid currentColor; }}
.chip.fresh {{ color: var(--fresh); background: var(--fresh-bg); }}
.chip.rotten {{ color: var(--rotten); background: var(--rotten-bg); }}
.chip.unknown {{ color: var(--ink-2); background: var(--muted-bg); }}
.prob {{ margin: 0; font-size: 0.8125rem; color: var(--ink-2); }}
.prob b {{ font-family: var(--font-data); color: var(--ink); font-weight: 600; }}
.rot {{ display: grid; grid-template-columns: auto minmax(0, 1fr) auto; gap: 8px; align-items: center; font-size: 0.75rem; color: var(--ink-2); }}
.meter {{ height: 8px; border-radius: 4px; background: var(--track); overflow: hidden; }}
.meter span {{ display: block; height: 100%; background: var(--rotten); }}
.rot-val {{ font-family: var(--font-data); font-variant-numeric: tabular-nums; }}
.note {{ margin: 0; font-size: 0.8125rem; border-radius: 6px; padding: 6px 8px; background: var(--muted-bg); color: var(--ink); }}
.note.warn {{ background: var(--warn-bg); color: var(--warn); }}
.heat-btn {{ justify-self: start; font: inherit; font-size: 0.8125rem; font-weight: 600; padding: 4px 10px; border-radius: 6px; cursor: pointer;
  background: var(--surface); color: var(--accent); border: 1px solid var(--accent); }}
.heat-btn[aria-pressed="true"] {{ background: var(--accent); color: var(--surface); }}
.empty {{ color: var(--ink-2); }}
.notes {{ border-top: 1px solid var(--line); padding-top: 14px; color: var(--ink-2); font-size: 0.8125rem; max-width: 80ch; margin: 0; padding-left: 18px; display: grid; gap: 4px; }}
@media (max-width: 420px) {{ .grid {{ grid-template-columns: 1fr 1fr; gap: 8px; }} .veg {{ font-size: 1.05rem; }} }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>채소 판별 리포트</h1>
    <div class="meta"><span>{generated}</span><span>사진 {total}장</span><span>{model}</span><span>{source}</span></div>
  </header>
  <section class="tiles" aria-label="요약 (누르면 그 사진만 보기)">{tiles}</section>
  <section class="tally" aria-label="채소별 판정">
    <h2>채소별 판정</h2>
    <div class="table-scroll"><table>
      <thead><tr><th scope="col">채소</th><th scope="col">정상</th><th scope="col">비정상</th><th scope="col">합계</th></tr></thead>
      <tbody>{table}</tbody>
    </table></div>
  </section>
  <div class="toolbar">
    <h2 id="listTitle">사진별 결과 · 확인이 필요한 사진부터</h2>
    <label class="switch"><input type="checkbox" id="heatAll"> 모든 사진에 상한 곳 표시</label>
  </div>
  <section class="grid" id="grid" aria-labelledby="listTitle">
{cards}
  </section>
  <p class="empty" id="empty" hidden>이 조건에 맞는 사진이 없습니다.</p>
  <ul class="notes">{notes}</ul>
</div>
<script>
(() => {{
  const cards = [...document.querySelectorAll(".card")], tiles = [...document.querySelectorAll(".tile")];
  const match = {{ all: () => true, rotten: (c) => c.dataset.verdict === "rotten", retake: (c) => c.dataset.retake === "1",
    unknown: (c) => c.dataset.verdict === "unknown" || c.dataset.verdict === "error", fresh: (c) => c.dataset.verdict === "fresh" }};
  tiles.forEach((t) => t.addEventListener("click", () => {{
    tiles.forEach((x) => x.setAttribute("aria-pressed", String(x === t)));
    let shown = 0;
    cards.forEach((c) => {{ const ok = match[t.dataset.filter](c); c.hidden = !ok; shown += ok; }});
    document.getElementById("empty").hidden = shown > 0;
  }}));
  const setHeat = (card, on) => {{
    const heat = card.querySelector(".heat"), photo = card.querySelector(".photo"), btn = card.querySelector(".heat-btn");
    if (!heat) return;
    heat.hidden = !on; photo.hidden = on; btn.setAttribute("aria-pressed", String(on));
    btn.textContent = on ? "원래 사진 보기" : "상한 곳 보기";
  }};
  cards.forEach((c) => {{ const b = c.querySelector(".heat-btn"); if (b) b.addEventListener("click", () => setHeat(c, b.getAttribute("aria-pressed") !== "true")); }});
  document.getElementById("heatAll").addEventListener("change", (e) => cards.forEach((c) => setHeat(c, e.target.checked)));
}})();
</script>
</body>
</html>
"""
