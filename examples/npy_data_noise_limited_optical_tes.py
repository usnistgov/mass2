import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl
    import pylab as plt
    import numpy as np
    import marimo as mo
    import mass2
    return mass2, np, pl, plt


@app.cell
def _(mass2, pl, pulse_traces):
    def channel_from_npy_files(pulses_traces, noise_traces, npresamples, frametime_s, ch_num: int = 0):
        header_df = pl.DataFrame()
        frametime_s = 1e-5
        df_noise = pl.DataFrame({"pulse": noise_traces})
        noise_ch = mass2.NoiseChannel(df_noise, header_df, frametime_s)
        nsamples, npulses = pulses_traces.shape
        header = mass2.ChannelHeader(
            "placeholder",
            data_source=None,
            ch_num=ch_num,
            frametime_s=frametime_s,
            n_presamples=npresamples,
            n_samples=nsamples,
            df=header_df,
        )
        df = pl.DataFrame({"pulse": pulse_traces + noise_traces})
        ch = mass2.Channel(df, header, npulses=npulses, noise=noise_ch)
        return ch
    return


@app.cell
def _(np):
    noisepath = "Ch3_60mV_50mK_46dB_noise_SRS.npy"
    pulsepath = "Ch3_60mV_50mK_46dB_pulses_SRS.npy"
    pulse_traces = np.load(pulsepath)[0].T*-10e4
    noise_traces = np.load(noisepath)[0].T*1e4
    return noise_traces, pulse_traces


@app.cell
def _(dummy_channel, noise_traces, pulse_traces):
    ch = dummy_channel(pulse_traces, noise_traces, npresamples = 300, frametime_s=6.25e-5)
    return (ch,)


@app.cell
def _(ch, plt):
    plt.figure()
    plt.plot(ch.df.limit(20)["pulse"].to_numpy().T)
    return


@app.cell
def _(ch):
    ch2 = ch.summarize_pulses()
    return (ch2,)


@app.cell
def _(ch2):
    ch2.df
    return


@app.cell
def _(ch2, np, plt):
    ch2.plot_hist("pulse_rms", np.linspace(0,10000,100))
    plt.gcf()
    return


@app.cell
def _(ch2, pl):
    ch3 = ch2.filter5lag(f_3db=1e5, use_expr=pl.col("pulse_rms").is_between(2000, 4000))
    return (ch3,)


@app.cell
def _(ch3, np, plt):
    ch3.plot_hist("5lagy", np.linspace(-1000,30000,1000))
    plt.gcf()
    return


@app.cell
def _(ch3, pl):
    ch4=ch3.rough_cal([0,800,1600], uncalibrated_col="5lagy", use_expr=pl.col("pulse_rms").is_between(-5000,30000), fwhm_pulse_height_units=750)
    return (ch4,)


@app.cell
def _(ch4, np, plt):
    ch4.plot_hist("energy_5lagy", np.linspace(-100,2000,1000))
    plt.gcf()
    return


@app.cell
def _(ch4, plt):
    ch4.step_plot(-1)
    plt.gcf()
    return


@app.cell
def _(ch4):
    ch4.steps
    return


@app.cell
def _(ch4, pl):
    avg_pulse = ch4.df.filter(pl.col("pulse_rms").is_between(-5000,30000)).to_numpy()
    return


@app.cell
def _(ch4, np, pl):
    np.isnan(ch4.df.filter(pl.col("pulse_rms").is_between(-5000,30000)).to_numpy())
    return


@app.cell
def _(np, pulse_traces):
    np.isnan(pulse_traces).any()
    return


app._unparsable_cell(
    r"""
    def pulse_rms_residual(pulse):
        filter = ch4.steps[1].filter
        a = np.dot(pulse, filter)
    
    ch4.with_column_map_step(f=)
    """,
    name="_"
)


if __name__ == "__main__":
    app.run()
