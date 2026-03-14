"""I/O helpers for reading/writing parquet and CSV."""

from __future__ import annotations

from pathlib import Path


def write_parquet(df, path: Path | str, **kwargs) -> Path:
    """Write a pandas or polars DataFrame to parquet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Handle polars
    if hasattr(df, "write_parquet"):
        df.write_parquet(str(path), **kwargs)
    else:
        # pandas
        df.to_parquet(path, engine="pyarrow", index=False, **kwargs)

    return path


def read_parquet(path: Path | str, columns: list[str] | None = None):
    """Read a parquet file, returning a pandas DataFrame."""
    import pandas as pd

    return pd.read_parquet(path, columns=columns, engine="pyarrow")
