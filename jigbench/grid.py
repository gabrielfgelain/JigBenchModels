"""Grid search of Andrade & Flesch (SPEC 2023, Tables IV and V) on our data.

Takes a base experiment and trains every combination of hidden layers (1, 2), neurons per
layer (1, 8, 16, 32), optimizer (Adam, RMSprop) and hidden-layer activation (linear, ReLU,
SELU, ELU). Configurations are ranked on the VALIDATION set; the test set is not used.

Run with `python -m jigbench.grid experiments/andrade_c.toml`; writes a single table,
outputs/grid/<name>.csv, instead of one folder per combination.
"""

import copy
import itertools
import sys
import tomllib
from pathlib import Path

import pandas as pd

from jigbench.baseline import load
from jigbench.features import OUTPUT_DIR
from jigbench.train import train

LAYERS = [1, 2]
NEURONS = [1, 8, 16, 32]
OPTIMIZERS = ["adam", "rmsprop"]
ACTIVATIONS = ["linear", "relu", "selu", "elu"]

if __name__ == "__main__":
    config_file = Path(sys.argv[1])
    with open(config_file, "rb") as f:
        base = tomllib.load(f)
    data = load()

    grid = list(itertools.product(OPTIMIZERS, ACTIVATIONS, LAYERS, NEURONS))
    rows = []
    for k, (optimizer, activation, layers, neurons) in enumerate(grid, 1):
        config = copy.deepcopy(base)
        config["model"].update(optimizer=optimizer, activation=activation,
                               hidden_layers=[neurons] * layers)
        run = train(config, data, verbose=False)
        val, tr = run["metrics"]["val"], run["metrics"]["train"]
        rows.append(dict(optimizer=optimizer, activation=activation, layers=layers,
                         neurons=neurons, epochs=run["epochs"],
                         train_R2=tr["R2"], train_MAE_pct=tr["MAE_pct"],
                         val_R2=val["R2"], val_MAE=val["MAE"], val_MAE_pct=val["MAE_pct"]))
        print(f"[{k:2d}/{len(grid)}] {optimizer:7s} {activation:6s} {layers}x{neurons:<2d}  "
              f"epochs {run['epochs']:5d}   val R2 {val['R2']:6.3f}  MAE {val['MAE_pct']:5.1f}%",
              flush=True)

    out = OUTPUT_DIR / "grid"
    out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows).sort_values("val_MAE_pct")
    table.to_csv(out / f"{config_file.stem}.csv", index=False, float_format="%.4f")
    print("\nBest 10 on validation:")
    print(table.head(10).to_string(index=False, float_format="%.3f"))
