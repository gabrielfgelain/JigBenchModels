"""Access to the local JigBenchDataset clone and the version being used."""

import subprocess
import tomllib
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = REPO_ROOT / "config.toml"


def dataset_root() -> Path:
    """Path of the JigBenchDataset clone, read from config.toml."""
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            "config.toml not found: copy config.example.toml to config.toml "
            "and set the dataset path."
        )
    with open(CONFIG_FILE, "rb") as f:
        root = Path(tomllib.load(f)["dataset"]["path"])
    if not (root / ".dvc").is_dir():
        raise FileNotFoundError(f"{root} is not a DVC repository (JigBenchDataset clone).")
    return root


def test_data_dir() -> Path:
    """Folder with the 1 s / 100 kS/s test CSVs."""
    return dataset_root() / "data" / "Test Data"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(dataset_root()), *args],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def dataset_version() -> dict:
    """Commit of the dataset in use, to be saved next to every result.

    Warns if the dataset has uncommitted changes, because then the commit
    alone does not identify the data.
    """
    commit = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain") != ""
    if dirty:
        warnings.warn(
            "JigBenchDataset has uncommitted changes: results will not be "
            "reproducible from the commit alone."
        )
    return {
        "dataset_commit": commit,
        "dataset_dirty": dirty,
        "dataset_date": _git("log", "-1", "--format=%cI"),
    }


if __name__ == "__main__":
    print("Dataset:", dataset_root())
    for key, value in dataset_version().items():
        print(f"{key}: {value}")
    print("CSVs:", sum(1 for _ in test_data_dir().rglob("*.csv")))
