from pathlib import Path

import numpy as np

TESTS = {'do you need something', 'what are you doing', 'what do you do', 'had your food',
         'help me', 'what you want', 'i got hurt', 'do not hurt me', 'can i help you',
         'you can do it', 'how can i help you', 'what did you tell him'}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Rewrite CSLTR sentence splits (DESTRUCTIVE without --force)")
    parser.add_argument("--force", action="store_true", help="Required to actually rewrite files")
    parser.add_argument("--npz-dir", default="data/processed/isl-csltr/landmarks")
    args = parser.parse_args()

    files = sorted(Path(args.npz_dir).glob("*.npz"))
    print(f"Would rewrite {len(files)} files with test sentences: {sorted(TESTS)}")
    if not args.force:
        print("Dry run (no changes). Pass --force to apply.")
        return
    n = 0
    for p in files:
        a = np.load(p, allow_pickle=True)
        sent = str(a["sentence"])
        sp = "test" if sent.lower() in TESTS else "train"
        np.savez_compressed(p, **{k: a[k] for k in a.files if k != "split"}, split=np.array(sp))
        n += 1
    print("sentences rewritten:", n, "| test sentences:", len(TESTS))


if __name__ == "__main__":
    main()
