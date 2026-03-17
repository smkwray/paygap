"""U.S. male/female earnings gap estimation using free public data."""

__version__ = "0.1.0"


def _patch_pandas_parquet_fallback() -> None:
    """Allow parquet-like IO in environments without pyarrow/fastparquet."""
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover
        return

    if getattr(pd, "_gender_gap_parquet_fallback", False):
        return

    orig_to_parquet = pd.DataFrame.to_parquet
    orig_read_parquet = pd.read_parquet

    def _to_parquet(self, path, *args, **kwargs):
        try:
            return orig_to_parquet(self, path, *args, **kwargs)
        except ImportError:
            return self.to_pickle(path)

    def _read_parquet(path, *args, **kwargs):
        try:
            return orig_read_parquet(path, *args, **kwargs)
        except ImportError:
            return pd.read_pickle(path)

    pd.DataFrame.to_parquet = _to_parquet
    pd.read_parquet = _read_parquet
    pd._gender_gap_parquet_fallback = True


_patch_pandas_parquet_fallback()
