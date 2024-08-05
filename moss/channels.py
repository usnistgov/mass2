from dataclasses import dataclass, field
import polars as pl
import pylab as plt
import numpy as np
import functools
import collections
import mass
import moss
import joblib

@dataclass(frozen=True)
class Channels:
    channels: collections.OrderedDict[int, moss.Channel]
    description: str

    @property
    def ch0(self):
        assert len(self.channels) > 0, "channels must be non-empty"
        for v in self.channels.values():
            return v

    @functools.cache
    def dfg(self, exclude="pulse"):
        # return a dataframe containing good pulses from each channel,
        # exluding "pulse" by default
        # and including columns "key" (to be removed?) and "ch_num"
        # the more common call should be to wrap this in a convenient plotter
        dfs = []
        for ch_num, channel in self.channels.items():
            df = channel.df.select(pl.exclude(exclude)).filter(channel.good_expr)
            # key_series = pl.Series("key", dtype=pl.Int64).extend_constant(key, len(df))
            assert ch_num == channel.header.ch_num
            ch_series = pl.Series("ch_num", dtype=pl.Int64).extend_constant(
                channel.header.ch_num, len(df)
            )
            dfs.append(df.with_columns(ch_series))
        return pl.concat(dfs)

    def linefit(
        self,
        line,
        col,
        use_expr=True,
        has_linear_background=False,
        has_tails=False,
        dlo=50,
        dhi=50,
        binsize=0.5,
    ):
        model = mass.get_model(line, has_linear_background=False, has_tails=False)
        pe = model.spect.peak_energy
        _bin_edges = np.arange(pe - dlo, pe + dhi, binsize)
        df_small = self.dfg().lazy().filter(use_expr).select(col).collect()
        bin_centers, counts = moss.misc.hist_of_series(df_small[col], _bin_edges)
        params = model.guess(counts, bin_centers=bin_centers, dph_de=1)
        params["dph_de"].set(1.0, vary=False)
        result = model.fit(
            counts, params, bin_centers=bin_centers, minimum_bins_per_fwhm=3
        )
        result.set_label_hints(
            binsize=bin_centers[1] - bin_centers[0],
            ds_shortname=self.description,
            unit_str="eV",
            attr_str=col,
            states_hint=f"{use_expr=}",
            cut_hint="",
        )
        return result

    def map(self, f, allow_throw=True):
        new_channels = collections.OrderedDict()
        for key, channel in self.channels.items():
            try:
                new_channels[key] = f(channel)
            except KeyboardInterrupt:
                raise
            except Exception as ex:
                if allow_throw:
                    raise
                print(f"{key=} {channel=} failed this step")
        return Channels(new_channels, self.description)

    def linefit_joblib(self, line, col, prefer="threads", n_jobs=4):
        def work(key):
            channel = self.channels[key]
            return channel.linefit(line, col)

        parallel = joblib.Parallel(
            n_jobs=n_jobs, prefer=prefer
        )  # its not clear if threads are better.... what blocks the gil?
        results = parallel(joblib.delayed(work)(key) for key in self.channels.keys())
        return results

    def __hash__(self):
        # needed to make functools.cache work
        # if self or self.anything is mutated, assumptions will be broken
        # and we may get nonsense results
        return hash(id(self))

    def __eq__(self, other):
        return id(self) == id(other)

    @classmethod
    def from_ljh_path_pairs(cls, pulse_noise_pairs, description):
        _channels = collections.OrderedDict()
        for pulse_path, noise_path in pulse_noise_pairs:
            channel = moss.Channel.from_ljh(pulse_path, noise_path)
            _channels[channel.header.ch_num] = channel
        return cls(_channels, description)
    
    @classmethod
    def from_off_paths(cls, off_paths, description):
        channels = {}
        for path in off_paths:
            ch = moss.Channel.from_off(mass.off.OffFile(path))
            channels[ch.header.ch_num] = ch
        return cls(channels, description)

    @classmethod
    def from_ljh_folder(cls, pulse_folder, noise_folder=None, limit=None):
        import os
        assert os.path.isdir(pulse_folder),f"{pulse_folder=} {noise_folder=}"
        if noise_folder is None:
            paths = moss.ljhutil.find_ljh_files(pulse_folder)
            pairs = ((path, None) for path in paths)
        else:
            assert os.path.isdir(noise_folder), f"{pulse_folder=} {noise_folder=}"
            pairs = moss.ljhutil.match_files_by_channel(pulse_folder, noise_folder, limit=limit)
        description = f"from_ljh_folder {pulse_folder=} {noise_folder=}"
        return cls.from_ljh_path_pairs(pairs, description)
    
    def get_experiment_state_df(self, experiment_state_path=None):
        if experiment_state_path is None:
            first_ch = next(iter(self.channels.values()))
            ljh_path = first_ch.header.df["Filename"][0]
            experiment_state_path = moss.ljhutil.experiment_state_path_from_ljh_path(ljh_path)
        df = pl.read_csv(experiment_state_path, new_columns=["unixnano", "state_label"])
        # _col0, _col1 = df.columns
        df_es = df.select(pl.from_epoch("unixnano", time_unit="ns").dt.cast_time_unit("us").alias("timestamp"))
        # strip whitespace from state_label column
        sl_series = df.select(pl.col("state_label").str.strip_chars()).to_series()
        df_es = df_es.with_columns(state_label = pl.Series(values=sl_series, dtype=pl.Categorical))
        return df_es
    
    def with_experiment_state_by_path(self, experiment_state_path=None):
        df_es = self.get_experiment_state_df(experiment_state_path)
        return self.with_experiment_state(df_es)

    def with_experiment_state(self, df_es):
        # this is not as performant as making use_exprs for states
        # and using .set_sorted on the timestamp column
        ch2s = {}
        for ch_num, ch in self.channels.items():
            ch2s[ch_num] = ch.with_experiment_state_df(df_es)
        return Channels(ch2s, self.description)    
    
    def with_steps_dict(self, steps_dict):
        ch2s = {}
        for ch_num, steps in steps_dict.items():
            ch = self.channels[ch_num]
            ch2 = ch.with_steps(steps)
            ch2s[ch_num] = ch2
        return Channels(ch2s, self.description+"\nfollowed some steps!!")
    
    def concat_data(self, other_data):
        # sorting here to show intention, but I think set is sorted by insertion order as
        # an implementation detail so this may not do anything
        ch_nums = sorted(list(set(self.channels.keys()).union(other_data.channels.keys())))
        channels2 = {}
        for ch_num in ch_nums:
            ch = self.channels[ch_num]
            other_ch = other_data.channels[ch_num]
            ch2 = ch.concat_ch(other_ch)
            channels2[ch_num] = ch2
        return moss.Channels(channels2, self.description+other_data.description)
    
    @classmethod
    def from_df(cls, df, frametime_s=np.nan, n_presamples=None, n_samples=None, description="from Channels.channels_from_df"):
        # requres a column named "ch_num" containing the channel number
        keys_df = df.partition_by(by=["ch_num"], as_dict=True)
        dfs = {keys[0]:df for (keys, df) in keys_df.items()}
        channels = {}
        for ch_num, df in dfs.items():
            channels[ch_num] = moss.Channel(df, header=moss.ChannelHeader(description="from df",
                        ch_num=ch_num, 
                        frametime_s=frametime_s, 
                        n_presamples=n_presamples, 
                        n_samples=n_samples, 
                        df=df))
        return Channels(channels, description)
        
        
