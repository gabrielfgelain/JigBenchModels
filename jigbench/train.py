"""Train a perceptron (MLP) with PyTorch, writing out every step of the training loop.

Run with `python -m jigbench.train experiments/<name>.toml`; writes
outputs/experiments/<name>/results.json and predictions.png. The test set is not used here.

At every epoch the error is computed on train AND validation, so the plot shows when the
model starts to memorize the training tests (train keeps falling, validation stops).
"""

import json
import sys
import tomllib
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

from jigbench.baseline import COLORS, evaluate, load, plot_predictions
from jigbench.features import OUTPUT_DIR

ACTIVATIONS = {"linear": nn.Identity, "relu": nn.ReLU, "sigmoid": nn.Sigmoid,
               "tanh": nn.Tanh, "elu": nn.ELU, "selu": nn.SELU}
OPTIMIZERS = {"adam": torch.optim.Adam, "rmsprop": torch.optim.RMSprop}


def build_model(n_inputs: int, hidden_layers: list, activation: str) -> nn.Sequential:
    """Layers in sequence: linear (weights) -> activation -> ... -> one linear output."""
    layers, size = [], n_inputs
    for n in hidden_layers:
        layers += [nn.Linear(size, n), ACTIVATIONS[activation]()]
        size = n
    layers.append(nn.Linear(size, 1))
    return nn.Sequential(*layers)


def train(config: dict, data, verbose: bool = True) -> dict:
    """Train one model from a config; returns metrics, loss history and predictions."""
    cfg, target, features = config["model"], config["data"]["target"], config["data"]["features"]
    torch.manual_seed(cfg["seed"])
    train, val = data[data.split == "train"], data[data.split == "val"]

    # 1. Scale inputs and target to mean 0 / std 1, using train statistics only,
    #    so every feature has the same weight at the start.
    x_mean, x_std = train[features].mean(), train[features].std()
    y_mean, y_std = train[target].mean(), train[target].std()

    def to_tensors(d):
        x = torch.tensor(((d[features] - x_mean) / x_std).values, dtype=torch.float32)
        y = torch.tensor(((d[target] - y_mean) / y_std).values, dtype=torch.float32)
        return x, y.unsqueeze(1)

    x_train, y_train = to_tensors(train)
    x_val, y_val = to_tensors(val)

    # 2. The model, the error to minimize (mean squared error) and the optimizer that
    #    adjusts the weights from that error (Adam by default).
    model = build_model(len(features), cfg["hidden_layers"], cfg["activation"])
    loss_fn = nn.MSELoss()
    optimizer = OPTIMIZERS[cfg.get("optimizer", "adam")](
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])

    # 3. Training loop: one epoch = one pass over all training tests, in small batches.
    history = {"train": [], "val": []}
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        order = torch.randperm(len(x_train))
        for i in range(0, len(order), cfg["batch_size"]):
            batch = order[i:i + cfg["batch_size"]]
            prediction = model(x_train[batch])              # forward: compute the output
            loss = loss_fn(prediction, y_train[batch])      # how wrong it is
            optimizer.zero_grad()
            loss.backward()                                 # how each weight affects the error
            optimizer.step()                                # adjust the weights a little

        model.eval()
        with torch.no_grad():                               # only measure, no adjusting
            history["train"].append(loss_fn(model(x_train), y_train).item())
            history["val"].append(loss_fn(model(x_val), y_val).item())
        line = (f"epoch {epoch:5d}   loss train {history['train'][-1]:.4f}   "
                f"val {history['val'][-1]:.4f}")
        if verbose and (epoch % cfg["print_every"] == 0 or epoch == 1):
            print(line, flush=True)

        # Early stopping (optional): stop when the validation loss improved less than
        # min_improvement (e.g. 1%) over the last `patience` epochs.
        patience = cfg.get("patience")
        if patience and epoch > patience:
            if history["val"][-1] > (1 - cfg["min_improvement"]) * history["val"][-1 - patience]:
                if verbose:
                    print(f"{line}   (stopped: val improved less than "
                          f"{100 * cfg['min_improvement']:.0f}% in {patience} epochs)", flush=True)
                break

    # 4. Predictions back in the target units and the same metrics as the baseline.
    def predict(d):
        x, _ = to_tensors(d)
        with torch.no_grad():
            return model(x).squeeze(1).numpy() * y_std + y_mean

    sets = {name: (d[target], predict(d)) for name, d in (("train", train), ("val", val))}
    metrics = {name: evaluate(*s) for name, s in sets.items()}
    return dict(metrics=metrics, history=history, sets=sets, epochs=len(history["val"]))


if __name__ == "__main__":
    config_file = Path(sys.argv[1])
    with open(config_file, "rb") as f:
        config = tomllib.load(f)
    cfg, target = config["model"], config["data"]["target"]

    run = train(config, load())
    metrics, history = run["metrics"], run["history"]
    for name, r in metrics.items():
        print(f"{name:5s}  MAE {r['MAE']:6.1f} ({r['MAE_pct']:4.1f}%)  R2 {r['R2']:.3f}")

    out = OUTPUT_DIR / "experiments" / config_file.stem
    out.mkdir(parents=True, exist_ok=True)
    results = json.loads((OUTPUT_DIR / "split.json").read_text())
    results.update(config=config, metrics=metrics, loss_history=history)
    (out / "results.json").write_text(json.dumps(results, indent=2))

    fig, (ax, ax_loss) = plt.subplots(1, 2, figsize=(11, 5))
    plot_predictions(ax, target, run["sets"], metrics, f"MLP {cfg['hidden_layers']}")
    for name, values in history.items():
        ax_loss.plot(range(1, len(values) + 1), values, color=COLORS[name], lw=2, label=name)
    ax_loss.set(yscale="log", xlabel="Epoch", ylabel="Loss (scaled MSE, log)",
                title="Training: error per epoch")
    ax_loss.grid(color="#e5e4e0", lw=0.6)
    ax_loss.spines[["top", "right"]].set_visible(False)
    ax_loss.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "predictions.png", dpi=150)
