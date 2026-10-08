"""Exit when the process that started this one is gone, so no pipeline tool outlives its demo.

`mass2-live-demo` sets MASS2_LIVE_PARENT_PID for the tools it starts and calls `exit_with_parent()` in each of
their processes before the tool begins (the simulator calls it itself).
Run by hand, without the variable, a tool runs as before.
"""

import os
import threading
import time

PARENT_ENV = "MASS2_LIVE_PARENT_PID"


def exit_with_parent(every_s: float = 1.0) -> None:
    """If started by a demo, check every `every_s` seconds that it is still this process's parent, and exit at once if not."""
    parent = os.environ.get(PARENT_ENV)
    if parent is None:
        return

    def watch() -> None:
        while os.getppid() == int(parent):
            time.sleep(every_s)
        os._exit(0)  # the parent is gone (killed, crashed): nothing will read what this process writes

    threading.Thread(target=watch, daemon=True).start()
