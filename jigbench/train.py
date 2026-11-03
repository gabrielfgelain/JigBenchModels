"""Train a perceptron (MLP) from an experiment file and evaluate it on train and validation.

Run with `python -m jigbench.train experiments/<name>.toml`; writes
outputs/experiments/<name>/results.json (config, dataset version, metrics) and
predictions.png. The test set is not used here.
"""

import json
import sys
import tomllib
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from jigbench.baseline import evaluate, load, plot_predictions
from jigbench.features import OUTPUT_DIR


def build_model(cfg: dict):
    """Features are scaled to mean 0 / std 1 (fitted on train only), then the MLP."""
    return make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=tuple(cfg["hidden_layers"]),
            activation=cfg["activation"],
            learning_rate_init=cfg["learning_rate"],
            alpha=cfg["alpha"],
            max_iter=cfg["max_iter"],
            random_state=cfg["seed"],
        ),
    )


if __name__ == "__main__":
    config_file = Path(sys.argv[1])
    with open(config_file, "rb") as f:
        config = tomllib.load(f)
    target, features = config["data"]["target"], config["data"]["features"]

    data = load()
    train, val = data[data.split == "train"], data[data.split == "val"]
    model = build_model(config["model"]).fit(train[features], train[target])

    sets = {name: (d[target], model.predict(d[features]))
            for name, d in (("train", train), ("val", val))}
    metrics = {name: evaluate(*s) for name, s in sets.items()}
    for name, r in metrics.items():
        print(f"{name:5s}  MAE {r['MAE']:6.1f} ({r['MAE_pct']:4.1f}%)  R2 {r['R2']:.3f}")

    out = OUTPUT_DIR / "experiments" / config_file.stem
    out.mkdir(parents=True, exist_ok=True)
    results = json.loads((OUTPUT_DIR / "split.json").read_text())
    results.update(config=config, iterations=model[-1].n_iter_, metrics=metrics)
    (out / "results.json").write_text(json.dumps(results, indent=2))

    fig, ax = plt.subplots(figsize=(5.5, 5))
    plot_predictions(ax, target, sets, metrics, f"MLP {config['model']['hidden_layers']}")
    fig.tight_layout()
    fig.savefig(out / "predictions.png", dpi=150)
