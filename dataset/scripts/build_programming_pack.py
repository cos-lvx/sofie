#!/usr/bin/env python3
"""Sloučí programming reasoning chains do `dataset/training/programming_pack.txt`.

Vstup do `train-core-memory --dataset dataset/training/programming_pack.txt`.
Distillates oddělené `\\n\\n---\\n\\n` (stejný formát jako law_pack).

Pořadí: alfabeticky podle stem (SOL-elt → SOL-gfx → SOL-iskra → SOL-vesna).
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CHAINS = REPO / "dataset/reasoning_chains/programming"
PACK = REPO / "dataset/training/programming_pack.txt"
TOKENIZER_PATH = Path("/home/lvx/Models/falcon-h1-1.5b-instruct/tokenizer.json")
SEPARATOR = "\n\n---\n\n"


def main() -> int:
    files = sorted(CHAINS.glob("SOL-*.md"))
    contents = [p.read_text(encoding="utf-8").rstrip() for p in files]
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
                print(f"  seq_len={sl:<4}: {chunks} chunků")
        except ImportError:
            print("  (tokenizers nejsou nainstalovány)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
