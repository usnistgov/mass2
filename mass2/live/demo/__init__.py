"""The demo environment: everything needed to show mass2.live without a running instrument.

simulate.py  replays pulsedata LJH files into a growing Arrow IPC stream, at a playback speed
datasets.py  the replayable pulsedata datasets, their detector-array layouts and saved recipes (recipes/)
launcher.py  mass2-live-demo: runs the simulator, mass2-live-apply, mass2-live-hist and the viewer together
visitors.py  gives each visitor a run of their own, at an address of its own
parent.py    a tool started by the demo exits when the demo is gone
export.py    mass2-live-export: records the real pipeline and writes a standalone, shareable replay page

Nothing in mass2.live (the core) or mass2.live.viewer imports from here.
"""
