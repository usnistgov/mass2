"""Live analysis: apply a saved mass2 recipe to pulse records as they are written to an Arrow IPC stream.

core/     the loop that writes the Arrow files (mass2-live-apply), and the histogram loop (mass2-live-hist)
viewer/   mass2-live-view: the web page and its server, reading the files the core writes
demo/     mass2-live-demo and mass2-live-export: simulator, demo datasets, a run per visitor, shareable page
"""
