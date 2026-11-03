"""Linear baseline: one physical feature per target, fitted on train, evaluated on validation.

- C: the 120 Hz ripple of the rectifier is proportional to P/C, so C is linear in P/A120.
- ESR: the switching ripple (~5 kHz) grows with the ESR, so ESR is fitted on Asw.

The test set is not used here. Run with `python -m jigbench.baseline`; writes
outputs/baseline.json and outputs/baseline.png (predicted vs real).
"""

import json

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

from jigbench.features import OUTPUT_DIR
from jigbench.split import assign

BASELINES = {
    "C_uF": "P_over_A120",
    "ESR_mOhm": "Asw",
}

LABELS = {"C_uF": "C (µF)", "ESR_mOhm": "ESR (mΩ)"}
COLORS = {"train": "#2a78d6", "val": "#eb6834"}


def load() -> pd.DataFrame:
    features = pd.read_csv(OUTPUT_DIR / "features.csv", dtype={"r_code": str})
    split = pd.read_csv(OUTPUT_DIR / "split.csv")
    features["split"] = assign(features, split)
    features["P_over_A120"] = features.power_W / features.A120
    return features


def evaluate(y_true, y_pred) -> dict:
    return dict(
        MAE=mean_absolute_error(y_true, y_pred),
        MAE_pct=100 * mean_absolute_error(y_true, y_pred) / y_true.mean(),
        R2=r2_score(y_true, y_pred),
    )


def plot_predictions(ax, target, sets, results):
    """Predicted vs real; points on the dashed line are perfect predictions."""
    for name, (y_true, y_pred) in sets.items():
        r2 = results[target][name]["R2"]
        ax.scatter(y_true, y_pred, s=14, alpha=0.5, color=COLORS[name], linewidths=0,
                   label=f"{name} (R² {r2:.2f})")
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "--", color="#52514e", lw=1, label="ideal")
    ax.set(xlim=lims, ylim=lims, aspect="equal",
           xlabel=f"Real {LABELS[target]}", ylabel=f"Predicted {LABELS[target]}",
           title=f"{LABELS[target]} from {results[target]['feature']}")
    ax.grid(color="#e5e4e0", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")


if __name__ == "__main__":
    data = load()
    train, val = data[data.split == "train"], data[data.split == "val"]
    results = json.loads((OUTPUT_DIR / "split.json").read_text())
    fig, axes = plt.subplots(1, len(BASELINES), figsize=(11, 5))
    for ax, (target, feature) in zip(axes, BASELINES.items()):
        model = LinearRegression().fit(train[[feature]], train[target])
        sets = {name: (d[target], model.predict(d[[feature]]))
                for name, d in (("train", train), ("val", val))}
        results[target] = dict(feature=feature, **{n: evaluate(*s) for n, s in sets.items()})
        for name in sets:
            r = results[target][name]
            print(f"{target:9s} <- {feature:12s} {name:5s}  "
                  f"MAE {r['MAE']:6.1f} ({r['MAE_pct']:4.1f}%)  R2 {r['R2']:.3f}")
        plot_predictions(ax, target, sets, results)
    (OUTPUT_DIR / "baseline.json").write_text(json.dumps(results, indent=2))
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "baseline.png", dpi=150)
