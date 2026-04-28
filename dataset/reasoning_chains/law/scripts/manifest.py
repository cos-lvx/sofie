#!/usr/bin/env python3
"""Regeneruje MANIFEST.md z reasoning_chains/law/.

Ověřuje 6-sekční šablonu, párování se zdroji v sources/law/, počítá slova,
kompresi a citace §§. Spouštět z root repozitáře:

    python3 dataset/reasoning_chains/law/scripts/manifest.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
CHAINS = REPO_ROOT / "dataset/reasoning_chains/law"
SOURCES = REPO_ROOT / "dataset/sources/law"
MANIFEST = CHAINS / "MANIFEST.md"
TOKENIZER_PATH = Path("/home/lvx/Models/falcon-h1-1.5b-instruct/tokenizer.json")


def load_tokenizer():
    """Vrátí tokenizer nebo None, pokud `tokenizers` nebo soubor chybí.

    Manifest nesmí selhat, když se generuje na stroji bez tokenizeru
    (CI, jiný dev box). Token sloupec se v takovém případě vynechá.
    """
    if not TOKENIZER_PATH.exists():
        return None
    try:
        from tokenizers import Tokenizer  # type: ignore
    except ImportError:
        return None
    return Tokenizer.from_file(str(TOKENIZER_PATH))

REQUIRED_SECTIONS = [
    "Zdroj",
    "Kontext",
    "Analytický flow",
    "Aplikovatelné principy",
    "Závěr",
    "Přenositelný pattern",
]

PATTERN_LABEL = {
    "P1": "lex specialis / kvalifikace",
    "P2": "teleologický výklad",
    "P3": "eurokonformní výklad",
    "P4": "analogie / mezery",
    "P5": "rebus sic stantibus",
    "P6": "ratio vs obiter / sjednocování",
    "P7": "překonání vlastní judikatury",
    "P8": "balancing principů",
    "P9": "důkazní břemeno",
    "P10": "procesně-hmotná hranice",
    "P11": "kauzalita / škoda",
    "P12": "výklad vůle",
}


def derive_source_name(stem: str) -> str:
    return stem.replace("NS-", "NS_", 1) + ".txt"


def analyze() -> list[dict]:
    tokenizer = load_tokenizer()
    out: list[dict] = []
    for md in sorted(CHAINS.glob("NS-*.md")):
        text = md.read_text(encoding="utf-8")
        headings = re.findall(r"^##\s+(.+?)\s*$", text, re.MULTILINE)
        title_m = re.search(r"^#\s+(.+?)\s*$", text, re.MULTILINE)
        title = title_m.group(1) if title_m else ""
        pattern_m = re.search(r"\(P(\d{1,2})\)", title)
        pattern = f"P{pattern_m.group(1)}" if pattern_m else "?"
        words = len(re.findall(r"\S+", text))
        para_cites = len(re.findall(r"§\s*\d+", text))
        tokens = (
            len(tokenizer.encode(text, add_special_tokens=False).ids)
            if tokenizer is not None
            else None
        )

        src_name = derive_source_name(md.stem)
        src_path = SOURCES / src_name
        src_words = (
            len(re.findall(r"\S+", src_path.read_text(encoding="utf-8")))
            if src_path.exists()
            else 0
        )

        out.append(
            {
                "file": md.name,
                "title": title,
                "pattern": pattern,
                "words": words,
                "src_words": src_words,
                "src_ok": src_path.exists(),
                "missing_sections": [s for s in REQUIRED_SECTIONS if s not in headings],
                "para_cites": para_cites,
                "tokens": tokens,
            }
        )
    return out


def render(rows: list[dict]) -> str:
    rows_sorted = sorted(rows, key=lambda r: (int(r["pattern"][1:]), r["file"]))
    lines: list[str] = []

    lines.append("# NS Corpus — Manifest reasoning chains")
    lines.append("")
    lines.append("> Vygenerováno automaticky z `dataset/reasoning_chains/law/`. Needitovat ručně.")
    lines.append("> Aktualizovat skriptem `dataset/reasoning_chains/law/scripts/manifest.py`.")
    lines.append("")

    ok_struct = sum(1 for r in rows if not r["missing_sections"])
    ok_src = sum(1 for r in rows if r["src_ok"])
    lines.append(
        f"**Stav:** {len(rows)} distillates, 6-sekční šablona u {ok_struct}/{len(rows)}, "
        f"párování se zdroji {ok_src}/{len(rows)}."
    )
    lines.append("")

    total_w = sum(r["words"] for r in rows)
    total_src_w = sum(r["src_words"] for r in rows)
    no_cite = [r for r in rows if r["para_cites"] == 0]
    nbsp = " "
    lines.append("## Souhrn")
    lines.append("")
    lines.append(f"- Distillates: **{len(rows)}**")
    lines.append(
        f"- Slov v distillates: **{total_w:,}** (průměr {total_w // len(rows)} / chain)".replace(
            ",", nbsp
        )
    )
    lines.append(
        f"- Slov v plných zněních: **{total_src_w:,}** (komprese ~{total_src_w / total_w:.1f}×)".replace(
            ",", nbsp
        )
    )
    lines.append(
        f"- Délka chain: min {min(r['words'] for r in rows)} — max {max(r['words'] for r in rows)} slov"
    )
    if no_cite:
        names = ", ".join(f"`{r['file'].replace('.md', '')}`" for r in no_cite)
        lines.append(f"- Distillate bez citace §§: **{len(no_cite)}** ({names})")
    if rows[0].get("tokens") is not None:
        total_t = sum(r["tokens"] for r in rows)
        toks = sorted(r["tokens"] for r in rows)
        lines.append(
            f"- Tokenů (Falcon-H1 tokenizer): **{total_t:,}** (min {toks[0]} — max {toks[-1]}, průměr {total_t // len(rows)})".replace(
                ",", nbsp
            )
        )
    lines.append("")

    patt = Counter(r["pattern"] for r in rows)
    lines.append("## Distribuce podle patternů")
    lines.append("")
    lines.append("| Pattern | Téma | Počet | Distillates |")
    lines.append("|---|---|---|---|")
    for p in sorted(patt.keys(), key=lambda x: int(x[1:])):
        files = [
            r["file"].replace(".md", "").replace("NS-", "")
            for r in rows_sorted
            if r["pattern"] == p
        ]
        lines.append(
            f"| {p} | {PATTERN_LABEL.get(p, '?')} | {patt[p]} | {', '.join(files)} |"
        )
    lines.append("")

    has_tokens = rows_sorted and rows_sorted[0].get("tokens") is not None
    lines.append("## Plný seznam")
    lines.append("")
    if has_tokens:
        lines.append("| Distillate | Pattern | Slov | Tokenů | Komprese | §§ |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for r in rows_sorted:
            comp = f"{r['src_words'] / r['words']:.1f}×" if r["words"] else "—"
            lines.append(
                f"| `{r['file'].replace('.md', '')}` | {r['pattern']} | {r['words']} | {r['tokens']} | {comp} | {r['para_cites']} |"
            )
    else:
        lines.append("| Distillate | Pattern | Slov | Komprese | §§ |")
        lines.append("|---|---|---:|---:|---:|")
        for r in rows_sorted:
            comp = f"{r['src_words'] / r['words']:.1f}×" if r["words"] else "—"
            lines.append(
                f"| `{r['file'].replace('.md', '')}` | {r['pattern']} | {r['words']} | {comp} | {r['para_cites']} |"
            )
    lines.append("")

    lines.append("## Review kandidáti pro Ondru")
    lines.append("")
    lines.append(
        "Filtry vybírají chains s mírným odchýlením od mediánu. Smyslem je dát review"
    )
    lines.append(
        "prioritu, ne vytknout chybu — chains mohou být legitimně delší/kratší podle"
    )
    lines.append("složitosti judikátu.")
    lines.append("")

    if no_cite:
        lines.append("### Bez citace §§")
        lines.append("")
        for r in no_cite:
            lines.append(
                f"- `{r['file'].replace('.md', '')}` ({r['pattern']} {PATTERN_LABEL.get(r['pattern'], '?')}) — 0 citací §§."
            )
        lines.append("")

    low_compression = sorted(rows, key=lambda r: r["src_words"] / max(r["words"], 1))[:3]
    lines.append("### Nejnižší komprese (distillate ≈ zdroj — možná příliš mnoho faktů)")
    lines.append("")
    for r in low_compression:
        comp = r["src_words"] / max(r["words"], 1)
        lines.append(
            f"- `{r['file'].replace('.md', '')}` ({r['pattern']}, {comp:.1f}×) — zdroj má jen {r['src_words']} slov; zvážit, zda chain destiluje *metodu*, nebo jen převypráví."
        )
    lines.append("")

    by_words = sorted(rows, key=lambda r: r["words"])
    lines.append("### Nejkratší 5 (template suggestoval 400–600, prakticky min 770)")
    lines.append("")
    for r in by_words[:5]:
        lines.append(
            f"- `{r['file'].replace('.md', '')}` ({r['pattern']}, {r['words']} slov) — {r['title']}"
        )
    lines.append("")

    lines.append("### Nejdelší 5")
    lines.append("")
    for r in by_words[-5:][::-1]:
        lines.append(
            f"- `{r['file'].replace('.md', '')}` ({r['pattern']}, {r['words']} slov) — {r['title']}"
        )
    lines.append("")

    lines.append("### Kandidáti na reklasifikaci patternu (z verifikace 2026-04-19)")
    lines.append("")
    lines.append(
        "- `NS-23Cdo2486-2020` (aktuálně P5) — procesní usnesení o odkladu vykonatelnosti,"
    )
    lines.append(
        "  ne hmotněprávní rozhodnutí o rebus sic stantibus. Zvážit reklasifikaci nebo"
    )
    lines.append("  vyřazení. Distillate má poznámku.")
    lines.append(
        "- `NS-23Cdo672-2021` (aktuálně P4) — vymezení rozsahu úřední odpovědnosti přes"
    )
    lines.append("  srovnání pozitivních/negativních precedentů; spíš P1 (kvalifikace).")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    rows = analyze()
    issues: list[str] = []
    for r in rows:
        if r["missing_sections"]:
            issues.append(f"{r['file']}: chybí sekce {r['missing_sections']}")
        if not r["src_ok"]:
            issues.append(f"{r['file']}: chybí zdroj v sources/law/")
    if issues:
        print("STRUCTURE ISSUES:", file=sys.stderr)
        for i in issues:
            print(f"  {i}", file=sys.stderr)

    MANIFEST.write_text(render(rows), encoding="utf-8")
    print(f"Wrote {MANIFEST.relative_to(REPO_ROOT)} ({MANIFEST.stat().st_size} B)")
    print(f"  rows: {len(rows)}, structure issues: {len(issues)}")

    json_path = MANIFEST.with_suffix(".json")
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  also wrote {json_path.relative_to(REPO_ROOT)} (machine-readable)")

    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
