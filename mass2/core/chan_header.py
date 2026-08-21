from dataclasses import dataclass, field
import polars as pl
import os


@dataclass(frozen=True)
class ChannelHeader:
    """Metadata about a Channel, of the sort read from file header."""

    description: str = ""  # filename or date/run number, etc
    data_source: str | None = None  # complete file path, if read from a file
    ch_num: int = 0
    frametime_s: float = 1e-3
    n_presamples: int = 0
    n_samples: int = 1000
    df: pl.DataFrame = field(default_factory=pl.DataFrame, repr=False)
    subframediv: int = 64
    pulse_data_sources: tuple[str | None, ...] | None = field(default=None, repr=False)
    noise_data_source: str | None = field(default=None, repr=False)

    def leaf_data_sources(self) -> tuple[str | None, ...]:
        """Leaf file paths behind this header's raw data, or `(data_source,)` if not itself a concatenation."""
        if self.pulse_data_sources is not None:
            return self.pulse_data_sources
        return (self.data_source,)

    @classmethod
    def from_ljh_header_df(cls, df: pl.DataFrame) -> "ChannelHeader":
        """Construct from the LJH header dataframe as returned by LJHFile.to_polars()"""
        try:
            filepath = df.item(0, "Filename")
        except pl.exceptions.ColumnNotFoundError:
            filepath = ""
        return cls(
            description=os.path.split(filepath)[-1],
            data_source=filepath,
            ch_num=df["Channel"][0],
            frametime_s=df["Timebase"][0],
            n_presamples=df["Presamples"][0],
            n_samples=df["Total Samples"][0],
            subframediv=df["Subframe divisions"][0],
            df=df,
        )
