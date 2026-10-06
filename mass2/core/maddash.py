import argparse
from pathlib import Path
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import polars as pl
import numpy as np
from typing import Any

# ----------------------------------------
# Command Line Arguments & Globals
# ----------------------------------------
parser = argparse.ArgumentParser(description="Live X-Ray Spectra Dashboard")
parser.add_argument("data_dir", type=str, help="Path to the directory containing state_spectra.arrow and channel_spectra.arrow")
args: argparse.Namespace = parser.parse_args()

# Safely resolve the provided directory path
DATA_DIR: Path = Path(args.data_dir).resolve()

# Pre-compute the energy x-axis (4000 bins) globally
X_ENERGY: np.ndarray = np.linspace(0.125, 999.875, 4000)

# Reusable button configuration for toggling y-axis scale
LOG_LINEAR_BUTTONS = [
    dict(
        type="buttons",
        direction="right",
        x=0.01,  # 1% off the left edge of the plotting area
        y=0.98,  # 2% down from the top edge of the plotting area
        xanchor="left",
        yanchor="top",  # Anchoring to the top forces the buttons to drop downward, avoiding the title
        showactive=True,
        bgcolor="#E5E5E5",
        font=dict(color="#000000"),
        bordercolor="#888888",
        buttons=[
            dict(args=[{"yaxis.type": "linear"}], label="Linear", method="relayout"),
            dict(args=[{"yaxis.type": "log"}], label="Log", method="relayout"),
        ],
    )
]

# Default layout for initialization
DEFAULT_LAYOUT = dict(
    xaxis=dict(title="Energy (eV)", range=[0, 1000]),
    yaxis=dict(title="Intensity", type="linear"),
    template="plotly_dark",
    uirevision="constant",  # Helps Plotly know to preserve trace isolation and zoom
    updatemenus=LOG_LINEAR_BUTTONS,
)

# Initialize the Dash app
app = dash.Dash(__name__)

# Application Layout
app.layout = html.Div([
    html.H1(f"Live X-Ray Spectra Dashboard ({DATA_DIR.name})", style={"font-family": "sans-serif"}),
    # Initialize graphs with the bounded layout so [0, 1000] is enforced on load
    dcc.Graph(id="state-spectra-graph", figure=go.Figure(layout=dict(title="State Spectra", **DEFAULT_LAYOUT))),
    dcc.Graph(id="channel-spectra-graph", figure=go.Figure(layout=dict(title="Channel Spectra", **DEFAULT_LAYOUT))),
    # Trigger updates every 5 seconds
    dcc.Interval(id="polling-interval", interval=5000, n_intervals=0),
])


# Callback to update both graphs on every interval tick
@app.callback(
    [Output("state-spectra-graph", "figure"), Output("channel-spectra-graph", "figure")],
    [Input("polling-interval", "n_intervals")],
    [State("state-spectra-graph", "figure"), State("channel-spectra-graph", "figure")],
)
def update_dashboard(n_intervals: int | None, state_fig: dict | None, chan_fig: dict | None) -> tuple[go.Figure, go.Figure]:
    # ----------------------------------------
    # 1. State Spectra Processing
    # ----------------------------------------
    state_file: Path = DATA_DIR / "state_spectra.arrow"
    df_state: pl.DataFrame = pl.read_ipc(state_file)
    fig_state: go.Figure = go.Figure()

    for row in df_state.iter_rows(named=True):
        state: str = str(row.get("state_label", "Unknown State"))
        spectra_data: list[float] = row.get("spectra", [])

        events: int = int(row.get("events", sum(spectra_data)))

        fig_state.add_trace(go.Scatter(x=X_ENERGY, y=spectra_data, mode="lines", name=f"{state} ({events:,} events)"))

    # If the user has interacted with the graph, state_fig['layout'] holds their custom zoom/scale.
    # We pass it straight back to perfectly preserve their viewport.
    if state_fig and "layout" in state_fig:
        fig_state.update_layout(**state_fig["layout"])
    else:
        fig_state.update_layout(title="State Spectra", **DEFAULT_LAYOUT)

    # ----------------------------------------
    # 2. Channel Spectra Processing
    # ----------------------------------------
    chan_file: Path = DATA_DIR / "channel_spectra.arrow"
    df_chan: pl.DataFrame = pl.read_ipc(chan_file).sort("channel_number")
    fig_chan: go.Figure = go.Figure()

    for row in df_chan.iter_rows(named=True):
        channel_num: Any = row.get("channel_number", "Unknown")
        chan_spectra_data: list[float] = row.get("spectra", [])

        events: int = int(row.get("events", sum(chan_spectra_data)))

        label_str = str(channel_num)
        trace_name = label_str if "Chan" in label_str else f"Chan {label_str}"

        fig_chan.add_trace(go.Scatter(x=X_ENERGY, y=chan_spectra_data, mode="lines", name=f"{trace_name} ({events:,} events)"))

    # Preserve client-side layout manipulations
    if chan_fig and "layout" in chan_fig:
        fig_chan.update_layout(**chan_fig["layout"])
    else:
        fig_chan.update_layout(title="Channel Spectra", **DEFAULT_LAYOUT)

    return fig_state, fig_chan


if __name__ == "__main__":
    app.run(debug=True)
