from mass2.core import ParquetCache
import polars as pl
from dataclasses import dataclass


# ------------------- Pytest -------------------
def test_cached_method(tmp_path):
    cache = ParquetCache(cache_dir=tmp_path, max_size_bytes=5_000_000, prune_every=100)

    @dataclass
    class CalStep:
        inputs: list
        call_count: int = 0

        def __parquet_cache_hash__(self):
            # Opt-in hash for caching
            return id(self)

        @cache.cached_method()
        def compute(self, df: pl.DataFrame) -> pl.DataFrame:
            self.call_count += 1
            return df.select([pl.col("a") * 2])

        def __hash__(self):
            return hash(str(self.inputs))  # only hash inputs so that call_count doesnt affect hash

    proc = CalStep(inputs=["a"])
    df = pl.DataFrame({"a": range(10)})
    df2 = df.with_columns(b=range(10))
    df3 = df.select(a=pl.col("a") * 4)

    assert cache._set_counter == 0

    # First call computes and caches
    result1 = proc.compute(df)
    assert proc.call_count == 1
    assert cache._set_counter == 1

    # Second call loads from cache
    result2 = proc.compute(df)
    assert proc.call_count == 1  # method should not be called again
    assert result1.equals(result2)
    assert cache._set_counter == 1

    result3 = proc.compute(df2)
    assert proc.call_count == 1
    assert result1.equals(result3)
    assert cache._set_counter == 1

    result4 = proc.compute(df3)
    assert proc.call_count == 2
    assert not result1.equals(result4)
    assert cache._set_counter == 2

    # Verify a Parquet file exists in the cache directory
    cached_files = list(tmp_path.glob("*.parquet"))
    assert len(cached_files) == 2
