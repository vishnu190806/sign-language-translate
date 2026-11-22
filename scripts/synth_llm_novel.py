import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.synth_continuous import is_novel, stitch_clips, to_isl_order

tok = AutoTokenizer.from_pretrained("models/qwen15b", trust_remote_code=True)
mdl = AutoModelForCausalLM.from_pretrained(
    "models/qwen15b", torch_dtype=torch.float16, device_map="cuda",
    trust_remote_code=True).eval()

inv = json.load(open("data/processed/isl-csltr/inventory.json"))
items = inv["items"]
real = [tuple(str(g).strip().upper() for g in i["glosses"])
        for i in items if i.get("kind") == "sentence" and i.get("glosses")]
seen = set(real)
examples = "\n".join(" ".join(s) for s in real[:12])

joint = json.load(open("data/processed/landmarks/joint_label_map.json"))
vocab_words = sorted({k.upper() for k in joint})
bank_paths: dict[str, list] = {}
for d in ["data/processed/landmarks/isl", "data/processed/landmarks/isl500",
          "data/processed/landmarks/isl-csltr/word_landmarks"]:
    for p in sorted(Path(d).glob("*.npz")):
        try:
            with np.load(p, allow_pickle=True) as a:
                lab = str(a["label"]).strip()
        except Exception:
            continue
        key = lab.split(".")[-1].strip().upper() if "." in lab else lab.strip().upper()
        if key in vocab_words:
            bank_paths.setdefault(key, []).append((str(p), d))

N = 30
made, attempts = 0, 0
manifest = []
out_dir = Path("data/processed/synthetic_llm")
out_dir.mkdir(parents=True, exist_ok=True)
while made < N and attempts < N * 10:
    attempts += 1
    msgs = [{"role": "system", "content": (
        "You write short ISL gloss sequences (2-4 WORDS, UPPER CASE, space separated). "
        "Use ONLY these words: " + " ".join(vocab_words[:60]) + ". "
        "Output ONE line, nothing else.")},
        {"role": "user", "content": f"Examples of style:\n{examples}\nWrite one NEW different sequence:"}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    ids = enc["input_ids"].to("cuda")
    with torch.inference_mode():
        out = mdl.generate(ids, max_new_tokens=20, do_sample=True, temperature=0.9,
                           pad_token_id=tok.eos_token_id)
    line = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip().upper()
    seq = to_isl_order([w for w in line.replace(",", " ").split() if w in vocab_words][:4])
    if len(seq) < 2 or not is_novel(seq, seen):
        continue
    bank = {}
    ok = True
    for g in seq:
        entries = bank_paths.get(g, [])
        if not entries:
            ok = False
            break
        import random
        p, _ = entries[random.randrange(len(entries))]
        with np.load(p, allow_pickle=True) as a:
            bank[g] = [(np.asarray(a["features"], dtype=np.float32), "mixed")]
    if not ok:
        continue
    feats, bounds, signers = stitch_clips(seq, bank, seed=made, speed_jitter=0.15)
    np.savez_compressed(out_dir / f"llm_{made:04d}.npz", features=feats, glosses=np.array(seq))
    manifest.append({"id": made, "glosses": seq,
                     "bounds": [[g, s, e] for g, s, e in bounds]})
    seen.add(tuple(seq))
    made += 1
    print(f"made {made}/{N}: {' '.join(seq)}", flush=True)
(out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
print(f"DONE {made} LLM-novel sentences")
