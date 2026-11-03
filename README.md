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

The data is not stored in this repository. It comes from JigBenchDataset, pinned to a
specific version so results can be reproduced.
