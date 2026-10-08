"""The core of mass2.live, which a real instrument runs. Every file it writes is Arrow IPC.

loop.py          THE CORE LOOPS: run_live_apply (mass2-live-apply): read the new raw records, mass2 applies each
                 channel's recipe, append the analyzed records; and run_live_hist (mass2-live-hist): histogram the
                 good analyzed records by channel and state in time slices, refit one line every few seconds
apply.py         the mass2 call that applies the recipes (Recipe.calc_from_df) to many channels' records
arrow_stream.py  write, and follow while it grows, an Arrow IPC stream file
histogram.py     the histograms and the histogrammer (mass2's hist_of_series)
states.py        the experiment states, read by mass2 from the growing DASTARD experiment_state.txt
fit.py           the line fit (mass2's line models) on the summed histograms
cli.py           the two command lines

Nothing here imports from mass2.live.viewer or mass2.live.demo.
"""
