from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, ".")

from src.text import detect_flags
from src.text.polish import LlamaCppPolishEngine, RulesOnlyEngine, TransformersPolishEngine


def f1(pred: str, ref: str) -> float:
    p = pred.lower().replace("?", "").replace(".", "").replace(",", "").split()
    r = ref.lower().replace("?", "").replace(".", "").replace(",", "").split()
    if not p or not r:
        return 0.0
    rc = {}
    for w in r:
        rc[w] = rc.get(w, 0) + 1
    ov = 0
    for w in p:
        if rc.get(w, 0) > 0:
            ov += 1
            rc[w] -= 1
    prec = ov / len(p)
    rec = ov / len(r)
    return 2 * prec * rec / max(prec + rec, 1e-9)


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--system", default=None)
    parser.add_argument("--system-file", default=None)
    parser.add_argument("--out", default="results/polish_eval.json")
    parser.add_argument("--rows-out", default=None)
    a = parser.parse_args()

    import src.text.polish as _polish_mod
    sys_text = a.system
    if a.system_file:
        sys_text = Path(a.system_file).read_text().strip()
    if sys_text:
        _polish_mod.SYSTEM_OVERRIDE = sys_text
        print(f"system override ({len(sys_text)} chars)")

    rows = json.loads(Path("results/gloss2text_rules.json").read_text())["rows"]
    lex = set()
    for r in rows:
        lex.update(w.lower() for w in r["glosses"])
    llm_path = sorted(Path("models/llm").glob("*.gguf"))
    tf_path = Path("models/qwen15b")
    if tf_path.exists() and any(tf_path.glob("*.safetensors")):
        eng: object = TransformersPolishEngine(str(tf_path), lex)
        print(f"engine: transformers {tf_path}")
    elif llm_path:
        eng = LlamaCppPolishEngine(llm_path[0], lex)
        print(f"engine: llama.cpp {llm_path[0].name}")
    else:
        eng = RulesOnlyEngine()
        print("engine: rules-only (no model found)")
    n = em = 0
    f1s = []
    uses = {}
    per_row = []
    import time

    t0 = time.time()
    for r in rows:
        final, how = eng.polish(r["glosses"], detect_flags(r["glosses"]))  # type: ignore[attr-defined]
        uses[how] = uses.get(how, 0) + 1
        n += 1
        em += final.rstrip(".?").lower() == r["sentence"].lower()
        f1s.append(f1(final, r["sentence"]))
        per_row.append({"glosses": r["glosses"], "ref": r["sentence"],
                        "out": final, "how": how})
    if n == 0:
        raise ValueError("No rows evaluated")
    dt = time.time() - t0
    out = {"n": n, "exact": em / n, "mean_f1": sum(f1s) / n,
           "uses": uses, "seconds_total": round(dt, 1),
           "seconds_per_sentence": round(dt / n, 2)}
    Path(a.out).write_text(json.dumps(out, indent=1))
    if a.rows_out:
        Path(a.rows_out).write_text(json.dumps(per_row, indent=1))
    print(f"exact: {em/n*100:.1f}% mean F1: {sum(f1s)/n*100:.1f}% uses={uses} {dt/n:.2f}s/sent")
    print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()
