from pyannotate_runtime import collect_types
import polars as pl
import pylab as plt
import numpy as np
import marimo as mo
import moss
import pulsedata
import mass
import pathlib

collect_types.init_types_collection()
with collect_types.collect():
    off_paths = moss.ljhutil.find_ljh_files(str(pulsedata.off["ebit_20240722_0006"]), ext=".off")
    off = mass.off.OffFile(off_paths[0])

    def from_off_paths(cls, off_paths):
        channels = {}
        for path in off_paths:
            ch = from_off(moss.Channel, mass.off.OffFile(path))
            channels[ch.header.ch_num] =ch
        return moss.Channels(channels, "from_off_paths")

    def from_off(cls, off):
        df = pl.from_numpy(off._mmap)
        df = df.select(pl.from_epoch("unixnano", time_unit="ns").dt.cast_time_unit("us").alias("timestamp")).with_columns(df).select(pl.exclude("unixnano"))
        header = moss.ChannelHeader(f"{off}",
                    off.header["ChannelNumberMatchingName"],
                    off.framePeriodSeconds,
        off._mmap["recordPreSamples"][0],
        off._mmap["recordSamples"][0],
        pl.DataFrame(off.header))
        ch = cls(df,
                        header)
        return ch
        
    data = from_off_paths(moss.Channels, off_paths)
    data


collect_types.dump_stats('type_info.json')