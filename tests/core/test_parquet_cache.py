from mass2.core import ParquetCache
from mass2.core import parquet_cache
import polars as pl
from dataclasses import dataclass


# ------------------- Pytest -------------------
def test_cached_method(tmp_path):
    cache = ParquetCache(cache_dir=tmp_path, max_size_bytes=5_000_000, prune_every=100)

    @dataclass
    class CalStepDummyForTest:
        inputs: list
        call_count: int = 0

        def _parquet_cache_hash(self):
            h = hash(str(self.inputs))  # only hash inputs so that call_count doesnt affect hash
            return h
        
        @cache.cached_method()
        def compute(self, df: pl.DataFrame) -> pl.DataFrame:
            assert set(self.inputs) == set(df.columns)
            self.call_count += 1
            return df.select([pl.col("a") * 2])

    step = CalStepDummyForTest(inputs=["a"])
    df = pl.DataFrame({"a": range(10)})
    df2 = df.with_columns(b=range(10))
    df3 = df.select(a=pl.col("a") * 4)
    key1 = cache.compute_key(step, df)

    assert cache._set_counter == 0

    # First call computes and caches
    result1 = step.compute(df)
    assert step.call_count == 1
    assert cache._set_counter == 1

    # Second call loads from cache
    result2 = step.compute(df)
    assert step.call_count == 1  # method should not be called again
    assert result1.equals(result2)
    assert cache._set_counter == 1

    key2 = cache.compute_key(step, df2.select(["a"]))
    assert key1 == key2
    assert parquet_cache.hash_df_by_sampling(df) == parquet_cache.hash_df_by_sampling(df2.select(["a"]))
    result3 = step.compute(df2.select(step.inputs))
    assert step.call_count == 1
    assert result1.equals(result3)
    assert cache._set_counter == 1

    result4 = step.compute(df3)
    assert step.call_count == 2
    assert not result1.equals(result4)
    assert cache._set_counter == 2

    # Verify a Parquet file exists in the cache directory
    cached_files = list(tmp_path.glob("*.parquet"))
    assert len(cached_files) == 2
