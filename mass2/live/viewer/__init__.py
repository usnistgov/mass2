"""The live viewer: a small web server (server.py) and the page it serves (viewer.html).

It reads only the files mass2-live-hist writes. Whoever runs the pipeline can plug in a
`Controller` (see server.py) to offer dataset switching and a speed control; mass2-live-demo does.
"""
