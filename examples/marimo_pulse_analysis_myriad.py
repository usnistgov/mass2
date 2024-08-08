import marimo

__generated_with = "0.7.17"
app = marimo.App(width="medium", app_title="MOSS intro")


@app.cell
def __(mo):
    mo.md(
        """
        #MOSS internals introdution
        MOSS is the Microcalorimeter Online Spectral Software, a replacement for MASS. MOSS support many algorithms for pulse filtering, calibration, and corrections. MOSS is built on modern open source data science software, including pola.rs and marimo. MOSS supports some key features that MASS struggled with including:
          * consecutive data set analysis
          * online (aka realtime) analysis
          * easily supporting different analysis chains
        """
    )
    return


@app.cell
def __():
    import polars as pl
    import pylab as plt
    import numpy as np
    import marimo as mo
    import pulsedata
    return mo, np, pl, plt, pulsedata


@app.cell
def __():
    import moss
    return moss,


@app.cell
def __(mo):
    mo.md(
        """
        # Load data
        Here we load the data, then we explore the internals a bit to show how MOSS is built.
        """
    )
    return


@app.cell
def __(moss, pulsedata):
    _p = pulsedata.pulse_noise_ljh_pairs["20240718"]
    data = moss.Channels.from_ljh_folder(
        pulse_folder=_p.pulse_folder, noise_folder=_p.noise_folder
    )
    data
    return data,


@app.cell
def __(mo):
    mo.md(
        """
        # basic analysis
        The variables `data` is the conventional name for a `Channels` object. It contains a list of `Channel` objects, conventinally assigned to a variable `ch` when accessed individualy. One `Channel` represents a single pixel, whiles a `Channels` is a collection of pixels, like a whole array.

        The data tends to consist of pulse shapes (arrays of length 100 to 1000 in general) and per pulse quantities, such as the pretrigger mean. These data are stored internally as pola.rs `DataFrame` objects. 

        The next cell shows a basic analysis on multiple channels. The function `data.transform_channels` takes a one argument function, where the one argument is a `Channel` and the function returns a `Channel`, `data.transform_channels` returns a `Channels`. There is no mutation, and we can't re-use variable names in a reactive notebook, so we store the result in a new variable `data2`.
        """
    )
    return


@app.cell
def __(data):
    data2 = data.map(
        lambda channel: channel.summarize_pulses()
        .with_good_expr_pretrig_mean_and_postpeak_deriv())
    return data2,


@app.cell
def __(data2, mo):
    _ch_nums = list(str(_ch_num) for _ch_num in data2.channels.keys())
    dropdown_ch = mo.ui.dropdown(
        options=_ch_nums, value=_ch_nums[0], label="channel number"
    )
    mo.md(f"These are the channels we have in our data: {dropdown_ch}")

    return dropdown_ch,


@app.cell
def __(data2, dropdown_ch, pl):
    ch_num = int(dropdown_ch.value)
    selected_ch = data2.channels[int(ch_num)]
    selected_ch.df.select(pl.exclude("pulse"))
    return ch_num, selected_ch


@app.cell
def __(mo, plt, selected_ch):
    plt.hist(selected_ch.df["min_value"], bins =1000)
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def __(mo, np, plt, selected_ch):
    trigger = 797
    trace = 300
    _pulse = selected_ch.df["pulse"][trace]
    _pulse2  = (selected_ch.df["pretrig_mean"][trace] -np.array(_pulse))/4096
    level_shift = round(np.mean(_pulse2[-100:-1]))
    plt.plot((_pulse2),'k.-')
    print (level_shift)
    # plt.plot((_pulse2),'r.-')
    # idx = np.where(np.diff(np.gradient(_pulse2)[::2]/4096)>=1)
    args = np.where(np.abs(np.diff((_pulse2)))>1)[0]
    level_shift = np.mean(_pulse2[-100:-1])
    phi0_shift = round(level_shift)
    # _pulse2[args[-1]+1::]-=phi0_shift*4096
    diffs = np.diff((_pulse2))[args]
    diffs = np.round(diffs)
    for _i, arg in enumerate(args):
        _pulse2[arg+1::]-=diffs[_i]
    # _pulse2[793::] = _pulse2[792] - _pulse2[793::]
    plt.plot(_pulse2,'r.-')
    # # print(diffs)
    # print(args)
    # # plt.plot(args, _pulse[args], '.')
    # plt.grid()

    mo.mpl.interactive(plt.gcf())
    return arg, args, diffs, level_shift, phi0_shift, trace, trigger


@app.cell
def __(pl, selected_ch):

    cut1 = selected_ch.df.filter((pl.col("pretrig_rms")<=50) & (pl.col("postpeak_deriv")<=20))
                                 
    cut1.select(pl.exclude("pulse"))

    return cut1,


@app.cell
def __():
    ch_to_ph_cut = {9217:3900, 9218: 3000, 9219:3670, 9220:2500, 9221:2000, 9223:3000, 9224:6000,9225:2000,
                    9226:2000, 9227: 7000, 9231:50, 9234:4000, 9235:435, 9237: 2150, 9239:1000, 9240:2000, 9241:3000}
    return ch_to_ph_cut,


@app.cell
def __(ch_num, ch_to_ph_cut):
    min_value_ll = ch_to_ph_cut[ch_num]


    return min_value_ll,


@app.cell
def __(min_value_ll):
    min_value_hl = min_value_ll +100
    return min_value_hl,


@app.cell
def __(cut1, min_value_hl, min_value_ll, mo, np, pl, plt):
    cut2 = cut1.filter((pl.col("min_value")<=int(min_value_hl))&(pl.col("min_value")>=int(min_value_ll)))
    # cut2 = cut1
    pulse = cut2["pulse"]
    pulse_bslnshift = ((np.ones_like(pulse).T*np.array(cut2["pretrig_mean"])).T -np.array(pulse))/4096
    # peak_pos = np.argmax(pulse_bslnshift, axis = 1)
    good_pulses = np.array([])
    # print(pulse[1,:].shape)
    for i in range(90):
        pulse_ = pulse_bslnshift[i]
        if np.round(pulse_[-1])==0:
            plt.plot(pulse_)
            good_pulses = np.append(good_pulses, np.array(pulse_), axis = 0)
        # plt.plot(peak_pos, pulse_bslnshift[i][peak_pos] , 'r*')

    # plt.plot(np.mean(cut2["pretrig_mean"]-pulse))
    mo.mpl.interactive(plt.gcf())
    return cut2, good_pulses, i, pulse, pulse_, pulse_bslnshift


@app.cell
def __(selected_ch):
    dt = selected_ch.header.frametime_s
    return dt,


@app.cell
def __():
    Min_SI = 249e-12#246e-12  # Henry = Weber/Amp
    phi0 = 2.069e-15  # Weber
    Min_phi0_per_amp = Min_SI / phi0  # phi0/Amp
    arbs_per_phi0 = 4096  # is 4096 when DASTARD is used
    amp_per_arb = 1 / Min_phi0_per_amp / arbs_per_phi0
    return Min_SI, Min_phi0_per_amp, amp_per_arb, arbs_per_phi0, phi0


@app.cell
def __(good_pulses, np):

    n_good = int(len(good_pulses)/4000)
    good_pulses_ = good_pulses.reshape((n_good,4000))
    mean = np.mean(good_pulses_, axis = 0)

    return good_pulses_, mean, n_good


@app.cell
def __(ch_num, dt, mean, min_value_hl, min_value_ll, mo, n_good, np, plt):

    # avg_pulse = np.mean(good, axis=0)
    time = np.arange(len(mean))*dt
    plt.plot(time*1e3, mean)
    plt.title(f'Average of {n_good}  pulses on channel {ch_num} in ph = [{min_value_ll},{min_value_hl}]')
    plt.xlabel('Time (ms)', fontsize  = 12)
    plt.ylabel(r'I$_{tes} ~ (\phi_0$)', fontsize = 12)
    mo.mpl.interactive(plt.gcf())
    return time,


@app.cell
def __(ch_num, mean, np, time):
    savePath = 'C:\\Users\\anr29\\OneDrive - NIST\\Data\\KPAC\\Pulses\\Raw data\\'
    np.savez(savePath+f'Channel{ch_num}_avgpulse.npz', col1 = time, col2 = mean)
    return savePath,


@app.cell
def __(cut2, pl):
    cut2.select(pl.exclude("pulse"))
    return


@app.cell
def __(ch_num, cut2, min_value_hl, min_value_ll, mo, plt, selected_ch):
    plt.hist(selected_ch.df["min_value"], bins = 1000)
    plt.hist(cut2["min_value"], bins = 10)
    plt.axvline(min_value_ll, color ='k', ls = '--', zorder = 0, lw =0.5)
    plt.axvline(min_value_hl, color ='k', ls = '--', zorder = 0, lw =0.5)
    # plt.xlim (3000,5000)
    plt.title(f'Raw vs selected pulses histogram for channel {ch_num}')

    mo.mpl.interactive(plt.gcf())
    return


if __name__ == "__main__":
    app.run()
