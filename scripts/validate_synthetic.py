import json

import numpy as np

man = json.load(open("data/processed/synthetic/manifest.json"))
inv = json.load(open("data/processed/isl-csltr/inventory.json"))
items = inv["items"]
real = {tuple(str(g).strip().upper() for g in i["glosses"])
        for i in items if i.get("kind") == "sentence" and i.get("glosses")}
syn = [tuple(m["glosses"]) for m in man]
print("synthetic:", len(syn), "| overlap with real:", len(set(syn) & real))
rb = set()
for s in real:
    for a, b in zip(s, s[1:]):
        rb.add((a, b))
sb = set()
for s in syn:
    for a, b in zip(s, s[1:]):
        sb.add((a, b))
print(f"real bigrams: {len(rb)}, synthetic: {len(sb)}, ratio: {len(sb) / max(len(rb), 1):.2f}x")
sg = set()
for m in man:
    sg.update(m["signers"])
print("signer sources:", len(sg))
ok = 0
for m in man[:20]:
    a = np.load(f"data/processed/synthetic/synth_{m['id']:04d}.npz", allow_pickle=True)
    t = 0
    good = True
    for (g, s, e) in m["bounds"]:
        if s != t:
            good = False
        t = e
    if good and t == a["features"].shape[0]:
        ok += 1
print(f"boundary-exact files: {ok}/20")
