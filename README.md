# JigBenchModels

Machine learning models to estimate DC-link capacitor health from the test bench
measurements in [JigBenchDataset](https://dagshub.com/gabrielfgelain/JigBenchDataset).

## Setup

Requires Python 3.11.

```bash
git clone <this-repo-url>
cd JigBenchModels
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-lock.txt
```

`requirements.txt` lists the main packages; `requirements-lock.txt` pins every package
(`pip freeze`) and is the one to install from to reproduce results exactly.

## Data

The data is not stored in this repository. It is read from a local clone of
JigBenchDataset:

1. Clone JigBenchDataset and run `dvc pull` (see its README).
2. Copy `config.example.toml` to `config.toml` and set `path` to that clone.
3. Check it:

   ```bash
   python -m jigbench.dataset
   ```

Every result records the dataset commit (`jigbench.dataset.dataset_version()`), so it can
be reproduced by checking out that commit in JigBenchDataset and running `dvc pull`.
A warning is shown if the dataset has uncommitted changes.
