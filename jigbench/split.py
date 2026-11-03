"""Train / validation / test split by capacitor, so the test set has capacitors never seen in training.

Run with `python -m jigbench.split` after `jigbench.features`; writes outputs/split.csv
(one row per capacitor group, with its split) and outputs/split.json (settings used).

Rules:
- The unit of the split is the capacitor (C, ESR), not the CSV: a capacitor goes to one
  split with all its tests, at every operating point.
- With 0 arms the 15 R codes give the same capacitor, so they form a single group. It goes
  to training, since it is the end of the C and ESR range.
- For 1 to 7 arms, each arm count has 15 R codes, split into N_VAL for validation, N_TEST
  for test and the rest for training. This keeps every C level in every split.
- The repeated reference tests are kept apart (split "ref"), for repeatability only.
"""

import json

import numpy as np
import pandas as pd

from jigbench.features import OUTPUT_DIR

SEED = 2026
N_VAL = 2
N_TEST = 2


def capacitor_group(arms: int, r_code: str) -> str:
    return "0Arms" if arms == 0 else f"{arms}Arms_R{r_code}"


def make_split(features: pd.DataFrame) -> pd.DataFrame:
    groups = (
        features.assign(group=[capacitor_group(a, r) for a, r in zip(features.arms, features.r_code)])
        .groupby("group", as_index=False)
        .agg(arms=("arms", "first"), C_uF=("C_uF", "first"), ESR_mOhm=("ESR_mOhm", "first"))
    )
    rng = np.random.default_rng(SEED)
    groups["split"] = "train"
    for arms in range(1, 8):
        idx = rng.permutation(groups.index[groups.arms == arms])
        groups.loc[idx[:N_VAL], "split"] = "val"
        groups.loc[idx[N_VAL:N_VAL + N_TEST], "split"] = "test"
    return groups.sort_values(["arms", "group"]).reset_index(drop=True)


def assign(features: pd.DataFrame, split: pd.DataFrame) -> pd.Series:
    """Split of each test (row of the feature table)."""
    group = [capacitor_group(a, r) for a, r in zip(features.arms, features.r_code)]
    s = pd.Series(group, index=features.index).map(split.set_index("group").split)
    return s.where(~features.is_ref, "ref")


if __name__ == "__main__":
    features = pd.read_csv(OUTPUT_DIR / "features.csv", dtype={"r_code": str})
    split = make_split(features)
    split.to_csv(OUTPUT_DIR / "split.csv", index=False)
    meta = json.loads((OUTPUT_DIR / "features.json").read_text())
    meta.update(seed=SEED, n_val_per_arms=N_VAL, n_test_per_arms=N_TEST)
    (OUTPUT_DIR / "split.json").write_text(json.dumps(meta, indent=2))

    tests = assign(features, split)
    print("capacitors:", split.split.value_counts().to_dict())
    print("tests:     ", tests.value_counts().to_dict())
