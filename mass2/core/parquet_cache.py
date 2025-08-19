# test_parquet_cache.py
from pathlib import Path
from dataclasses import dataclass, field
from functools import wraps
import polars as pl
import shutil

SEED = 42


def hash_df_by_sampling(df: pl.DataFrame) -> str:
    if len(df) == 0:
        return str(df.columns)
    n_sample = min(len(df), 10)
    return str(df.sample(n_sample, seed=SEED).hash_rows(seed=SEED).sum())


@dataclass
class ParquetCache:
    """
    ParquetCache for functions with signature (self: CalStep, df: pl.DataFrame).
    Only caches columns in step.inputs, hashes using sampling.
    """

    cache_dir: str = Path.cwd() / "_parquet_cache"
    max_size_bytes: int = 1_000_000_000
    prune_every: int = 100
    _set_counter: int = field(default=0, init=False, repr=False)

    def __post_init__(self):
        cache_path = Path(self.cache_dir)
        if cache_path.exists():  # we never want to re-use a directory
            shutil.rmtree(cache_path)  # delete old cache
        cache_path.mkdir(parents=True, exist_ok=True)  # recreate empty cache

    def compute_key(self, step, df: pl.DataFrame) -> str:
        # select only the columns needed
        key_str = str(hash((hash_df_by_sampling(df), step.parquet_cache_hash())))
        return key_str

    def _file_path(self, key_hash: str) -> Path:
        return Path(self.cache_dir) / f"{key_hash}.parquet"

    def set(self, key_hash: str, df: pl.DataFrame):
        path = self._file_path(key_hash)
        df.write_parquet(path)
        self._set_counter += 1
        if self._set_counter % self.prune_every == 0:
            self._enforce_size_limit()

    def get(self, key_hash: str) -> pl.DataFrame | None:
        path = self._file_path(key_hash)
        if path.exists():
            return pl.read_parquet(path)
        return None

    def _enforce_size_limit(self):
        files = list(Path(self.cache_dir).glob("*.parquet"))
        file_info = [(f, f.stat().st_size, f.stat().st_mtime) for f in files]
        total_size = sum(s for _, s, _ in file_info)
        if total_size <= self.max_size_bytes:
            return
        file_info.sort(key=lambda x: x[2])
        for f, s, _ in file_info:
            try:
                f.unlink()
            except FileNotFoundError:
                continue
            total_size -= s
            if total_size <= self.max_size_bytes:
                break

    # ---------------- Decorator ----------------
    def cached_method(self):
        """Decorator for methods with signature (self: CalStep, df: pl.DataFrame)."""

        def decorator(method):
            @wraps(method)
            def wrapper(step, df: pl.DataFrame, *args, **kwargs):
                key = self.compute_key(step, df)

                cached = self.get(key)
                if cached is not None:
                    return cached

                result = method(step, df, *args, **kwargs)
                if not isinstance(result, pl.DataFrame):
                    raise TypeError("cached_method only supports DataFrame return values")
                self.set(key, result)
                return result

            return wrapper

        return decorator
