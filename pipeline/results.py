"""Read and write results/{target_id}.parquet (spec 6.4) and the matrix view of a results table."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.contract import Feature


def read_results(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_results(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)


@dataclass(frozen=True)
class Matrices:
    """A results table as numeric matrices, which is what scoring and the API work on."""

    keys: list[str]
    s: np.ndarray  # n_cells x n_features
    q: np.ndarray  # n_cells x n_features

    def index(self, key: str) -> int:
        return self.keys.index(key)

    def weights(self, weights: dict[str, float]) -> np.ndarray:
        return np.array([float(weights.get(key, 0.0)) for key in self.keys])

    def gates(self, gates: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
        """Return (gate similarities n_cells x n_gates, thresholds n_gates)."""
        keys = [key for key in gates if key in self.keys]
        columns = [self.s[:, self.index(key)] for key in keys]
        gate_s = np.stack(columns, axis=1) if columns else np.empty((self.s.shape[0], 0))
        return gate_s, np.array([gates[key] for key in keys])


def to_matrices(table: pd.DataFrame, features: dict[str, Feature]) -> Matrices:
    keys = list(features)
    s = np.column_stack([table[f"s_{key}"].to_numpy(dtype=np.float64) for key in keys])
    q = np.column_stack([table[f"q_{key}"].to_numpy(dtype=np.float64) for key in keys])
    return Matrices(keys=keys, s=s, q=q)
