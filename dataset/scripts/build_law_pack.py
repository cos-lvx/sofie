#!/usr/bin/env python3
"""Sloučí 43 reasoning chains do `dataset/training/law_pack.txt`.

Vstup do `train-core-memory --dataset dataset/training/law_pack.txt`.
Distillates oddělené `\\n\\n---\\n\\n`, aby tokenizer viděl výraznou
hranici mezi judikáty a model nemíchal patterny napříč chunky.

Pořadí: stejné jako v MANIFEST.md (P1 → P12, abecedně v rámci patternu),
aby při deterministic shuffle (seed=0) měl Ondra reprodukovatelný výsledek.

Spustit:
    python3 dataset/scripts/build_law_pack.py
    # nebo s tokenizací:
    uv run --with tokenizers python3 dataset/scripts/build_law_pack.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CHAINS = REPO / "dataset/reasoning_chains/law"
PACK = REPO / "dataset/training/law_pack.txt"
TOKENIZER_PATH = Path("/home/lvx/Models/falcon-h1-1.5b-instruct/tokenizer.json")
SEPARATOR = "\n\n---\n\n"


def pattern_key(stem: str) -> tuple[int, str]:
    """Stem `NS-31Cdo4356-2008` → (1, 'NS-31Cdo4356-2008'), kde 1 = P1."""
    text = (CHAINS / f"{stem}.md").read_text(encoding="utf-8")
    m = re.search(r"\(P(\d{1,2})\)", text)
    return (int(m.group(1)) if m else 99, stem)


def main() -> int:
    files = sorted([p.stem for p in CHAINS.glob("NS-*.md")], key=pattern_key)
    contents = [(CHAINS / f"{stem}.md").read_text(encoding="utf-8").rstrip() for stem in files]
    pack = SEPARATOR.join(contents) + "\n"
    PACK.write_text(pack, encoding="utf-8")
    chars = len(pack)
    words = len(re.findall(r"\S+", pack))
    print(f"Wrote {PACK.relative_to(REPO)}")
    print(f"  files concatenated: {len(files)}")
    print(f"  chars: {chars:,}".replace(",", " "))
    print(f"  words: {words:,}".replace(",", " "))

    if TOKENIZER_PATH.exists():
        try:
            from tokenizers import Tokenizer  # type: ignore

            tok = Tokenizer.from_file(str(TOKENIZER_PATH))
            ids = tok.encode(pack, add_special_tokens=False).ids
            n = len(ids)
            print(f"  tokens (Falcon-H1): {n:,}".replace(",", " "))
            for sl in (1024, 2048, 4096):
                chunks = n // sl
                print(f"  seq_len={sl:<4}: {chunks} chunků (drop {n - chunks * sl} tail tokenů)")

            # Kolik patternů by se vešlo do jednoho chunku 4096?
            # Per-distillate token counts:
            per_file = []
            for stem in files:
                t = len(
                    tok.encode(
                        (CHAINS / f"{stem}.md").read_text(encoding="utf-8"),
                        add_special_tokens=False,
                    ).ids
                )
                per_file.append((stem, t))
            sep_tokens = len(tok.encode(SEPARATOR, add_special_tokens=False).ids)
            print(f"  separator overhead: {sep_tokens} tokenů × {len(files) - 1} = {sep_tokens * (len(files) - 1)}")
            longest = max(per_file, key=lambda x: x[1])
            print(f"  longest distillate: {longest[0]} ({longest[1]} tokenů)")
            if longest[1] > 2048:
                print("  WARN: nejdelší distillate > seq_len 2048 — při seq_len=2048 dojde k cross-distillate splitu")
        except ImportError:
            print("  (tokenizers nejsou nainstalovány — pro token count: uv run --with tokenizers ...)")
    else:
        print(f"  (tokenizer not found at {TOKENIZER_PATH})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
