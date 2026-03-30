import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Some examples of studying optical TES data from numpy files
    """)
    return


@app.cell
def _():
    import polars as pl
    import pylab as plt
    import numpy as np
    import marimo as mo
    import mass2
    import pulsedata

    return mass2, mo, np, pl, plt, pulsedata


@app.cell
def _(mass2, pulsedata):
    pulse_noise_pair = pulsedata.numpy["noise_limited_optical_tes"]
    noisepath = pulse_noise_pair.noise
    pulsepath = pulse_noise_pair.pulse
    ch = mass2.Channel.from_numpy(
        samplerate = 16e3,
        npresamples=300,
        pulse_fname=pulsepath,
        noise_fname=noisepath,
        description="from npy arrays",
        invert_data=True,
        row_index=True,
        rescale=1e4
    )
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
    ch2.plot_hist("peak_value", np.linspace(0, 5000, 100))
    plt.gcf()
    return


@app.cell
def _(ch2, pl):
    ch3 = ch2.filter5lag(f_3db=1e5, use_expr=pl.col("peak_value").is_between(1000, 1500))
    return (ch3,)


@app.cell
def _(ch3, np, plt):
    ch3.plot_hist("5lagy", np.linspace(-1000, 3000, 500))
    plt.gcf()
    return


@app.cell
def _(ch3):
    ch4 = ch3
    # here we would do calibration, but the mass2 gain based calibration fails from learning from points at energy=0 currently,
    # since gains=ph/e = inf when e=0
    return (ch4,)


@app.cell
def _(ch4, plt):
    ch4.step_plot(-1)
    plt.gcf()
    return


@app.cell
def _(ch4, pl):
    avg_pulse = ch4.df.filter(pl.col("5lagy").is_between(1000, 1500))["pulse"].to_numpy().mean(axis=0)
    return (avg_pulse,)


@app.cell
def _(avg_pulse, plt):
    plt.figure()
    plt.plot(avg_pulse)
    plt.gcf()
    return


@app.cell
def _(avg_pulse, np):
    def create_pulse_processor(avg_pulse):
        norm = np.linalg.norm(avg_pulse)
        unit_vec = avg_pulse / norm

        def get_residual_rms(pulse):
            height = np.dot(pulse, unit_vec)
            residuals = pulse - (height * unit_vec)
            return np.sqrt(np.mean(residuals**2))

        return get_residual_rms, unit_vec

    get_residual_rms, avg_pulse_normalized = create_pulse_processor(avg_pulse)
    return (get_residual_rms,)


@app.cell
def _(ch4, get_residual_rms, mass2, pl):
    ch5: mass2.Channel = ch4.with_column_map_step(f=get_residual_rms, input_col="pulse", output_col="residual_rms")
    ch5 = ch5.with_categorize_step({"clean": pl.lit(True), "residual_rms>100": pl.col(
        "residual_rms") > 100, "residual_rms>150": pl.col("residual_rms") > 150, "residual_rms>200": pl.col("residual_rms") > 200})
    return (ch5,)


@app.cell
def _(ch5: "mass2.Channel"):
    ch5.df
    return


@app.cell
def _(ch5: "mass2.Channel", pl, plt):
    plt.figure()
    plt.title("rejected pulses based on residual_rms>200")
    plt.plot(ch5.df.filter(pl.col("category") == "residual_rms>200").limit(100)["pulse"].to_numpy().T)
    plt.gcf()
    return


@app.cell
def _(ch5: "mass2.Channel", pl, plt):
    plt.figure()
    plt.title("rejected pulses based on residual_rms>150")
    plt.plot(ch5.df.filter(pl.col("category") == "residual_rms>150").limit(100)["pulse"].to_numpy().T)
    plt.gcf()
    return


@app.cell
def _():
    return


@app.cell
def _(ch5: "mass2.Channel", np, plt):
    ch5.plot_hist("residual_rms", np.arange(0, 400, 1))
    plt.gcf()
    return


@app.cell
def _(ch5: "mass2.Channel", plt):
    ch5.plot_scatter("index", "pretrig_mean", color_col="category")
    plt.gcf()
    return


@app.cell
def _(ch5: "mass2.Channel", plt):
    ch5.plot_scatter("index", "5lagy", color_col="category")
    plt.ylim(1000, 1500)
    plt.gcf()
    return


@app.cell
def _(ch5: "mass2.Channel", plt):
    ch5.plot_scatter("5lagx", "5lagy", color_col="category")
    plt.ylim(1000, 1500)
    plt.xlim(-1, 1)
    plt.gcf()
    return


@app.cell
def _(ch5: "mass2.Channel", np, plt):
    ch5.plot_hists("5lagy", np.linspace(-500, 3000, 1000), group_by_col="category")
    plt.yscale("log")
    plt.gcf()
    return


if __name__ == "__main__":
    app.run()
