import marimo

__generated_with = "0.7.17"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import mass
    import mass
    from mass.off import (
        ChannelGroup,
        getOffFileListFromOneFile,
        Channel,
        labelPeak,
        labelPeaks,
    )
    import numpy as np
    import pylab as plt
    import pulsedata
    return (
        Channel,
        ChannelGroup,
        getOffFileListFromOneFile,
        labelPeak,
        labelPeaks,
        mass,
        mo,
        np,
        plt,
        pulsedata,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        # why a mass.off style analysis?
        This is included in moss to 
        1. show how to use variable assignments to force evaluation order to handle the mutable nature of mass
        2. Show a direct 1-1 comparison of achieved resolution at various lines.
        """
    )
    return


@app.cell
def __(
    Channel,
    ChannelGroup,
    getOffFileListFromOneFile,
    mass,
    np,
    pulsedata,
):
    """
    Day Summary: Mostly Ne-like data, of W, Re, and Os

                 Less contamination of calibration than in 0722 due to weaker signal.
    """

    a = 1  # for marimo order
    folder_path = pulsedata.off["ebit_20240723_0000"]
    date = "20240723"
    orig_esf_path = folder_path / f"{date}_run0000_experiment_state.txt"
    off_path = str(folder_path / f"{date}_run0000_chan1.off")
    timing_path = folder_path / f"time_{date}.txt"

    esf = mass.off.ExperimentStateFile(orig_esf_path, excludeStates=[])

    data: ChannelGroup = ChannelGroup(
        getOffFileListFromOneFile(off_path, maxChans=400), experimentStateFile=esf
    )
    ds: Channel = data[1]
    data.learnResidualStdDevCut()
    states = ["START"]
    ds.calibrationPlanInit("filtValue")
    ds.calibrationPlanAddPoint(6020, "ZnLAlpha", states=states)
    ds.calibrationPlanAddPoint(7020, "GeLAlpha", states=states)
    ds.calibrationPlanAddPoint(8730, "AlKAlpha", states=states)
    ds.calibrationPlanAddPoint(22540, "ScKAlpha", states=states)
    ds.calibrationPlanAddPoint(26820, "VKAlpha", states=states)
    ds.calibrationPlanAddPoint(31290, "MnKAlpha", states=states)
    ds.calibrationPlanAddPoint(33640, "FeKAlpha", states=states)
    ds.calibrationPlanAddPoint(43237, "ZnKAlpha", states=states)
    ds.calibrationPlanAddPoint(48190, "GeKAlpha", states=states)


    data.alignToReferenceChannel(ds, "filtValue", np.arange(0, 60000, 10))
    data.learnPhaseCorrection(
        indicatorName="filtPhase",
        uncorrectedName="filtValue",
        correctedName="filtValuePC",
        states=states,
    )
    data.learnDriftCorrection(
        indicatorName="pretriggerMean",
        uncorrectedName="filtValuePC",
        correctedName="filtValuePCDC",
        states=states,
    )
    data.calibrateFollowingPlan("filtValuePCDC", "energy")
    ds.diagnoseCalibration()
    return (
        a,
        data,
        date,
        ds,
        esf,
        folder_path,
        off_path,
        orig_esf_path,
        states,
        timing_path,
    )


@app.cell
def __(a, ds, mo, plt):
    a
    ds.diagnoseCalibration()
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def __(a, ds, mo, plt):
    a  # for marimo order
    ds.diagnoseCalibration()
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def __(a, ds, mo, plt):
    a
    ds.plotAvsB("relTimeSec", "energy")
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def __(a, ds, mo, plt):
    a
    ds.plotAvsB("pretriggerMean", "energy")
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def __(a, ds, mo, plt):
    a
    ds.plotAvsB("filtPhase", "energy", states="START")
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def __(a, ds, mo, np, plt):
    a
    ds.plotHist(np.arange(0, 60000, 10), "filtValue", coAddStates=False)
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def __(a, ds, mo, plt):
    a
    result = ds.linefit("AlKAlpha", states="START")
    result.plotm()
    mo.mpl.interactive(plt.gcf())
    return result,


@app.cell
def __(a, data, np, plt):
    b=a
    data.calcExternalTriggerTiming()
    data._externalTriggerSubframes()
    plt.figure()
    plt.plot(np.diff(data._externalTriggerSubframes()[:10000]),".")
    plt.title("external trigger differences, this should all be one value!!")
    return b,


if __name__ == "__main__":
    app.run()
