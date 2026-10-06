import argparse
from pathlib import Path
import dash
from dash import dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
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

DEFAULT_LAYOUT: dict[str, Any] = dict(
    xaxis=dict(title="Energy (eV)", range=[0, 1000]),
    yaxis=dict(title="Intensity", type="linear"),
    template="plotly_dark",
    uirevision="constant",  # Helps Plotly know to preserve trace isolation and zoom
    updatemenus=LOG_LINEAR_BUTTONS,
)

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1(f"Live X-Ray Spectra Dashboard ({DATA_DIR.name})", style={"font-family": "sans-serif"}),
    # Hidden store to track the last modified times known to the client
    dcc.Store(id="client-version", data=""),
    dcc.Graph(id="total-spectra-graph", figure=go.Figure(layout=dict(title="Overall Spectrum (All Channels)", **DEFAULT_LAYOUT))),
    dcc.Graph(id="state-spectra-graph", figure=go.Figure(layout=dict(title="State Spectra", **DEFAULT_LAYOUT))),
    dcc.Graph(id="channel-spectra-graph", figure=go.Figure(layout=dict(title="Channel Spectra", **DEFAULT_LAYOUT))),
    # Fast polling (500ms). Costs almost nothing because of PreventUpdate
    dcc.Interval(id="polling-interval", interval=500, n_intervals=0),
])


@app.callback(
    [
        Output("total-spectra-graph", "figure"),
        Output("state-spectra-graph", "figure"),
        Output("channel-spectra-graph", "figure"),
        Output("client-version", "data"),
    ],
    [Input("polling-interval", "n_intervals")],
    [
        State("total-spectra-graph", "figure"),
        State("state-spectra-graph", "figure"),
        State("channel-spectra-graph", "figure"),
        State("client-version", "data"),
    ],
)
def update_dashboard(
    n_intervals: int | None, total_fig: dict | None, state_fig: dict | None, chan_fig: dict | None, client_version: str
) -> tuple[go.Figure, go.Figure, go.Figure, str]:
    state_file: Path = DATA_DIR / "state_spectra.arrow"
    chan_file: Path = DATA_DIR / "channel_spectra.arrow"

    # ----------------------------------------
    # Stateless Modification Check
    # ----------------------------------------
    try:
        # Get modification times (st_mtime). Default to 0.0 if file doesn't exist yet.
        state_mtime = state_file.stat().st_mtime if state_file.exists() else 0.0
        chan_mtime = chan_file.stat().st_mtime if chan_file.exists() else 0.0
    except OSError:
        # Failsafe in case a file is caught exactly mid-deletion during atomic rename
        raise PreventUpdate

    current_server_version = f"{state_mtime}_{chan_mtime}"

    if current_server_version == client_version:
        raise PreventUpdate

    # ----------------------------------------
    # 1. State and Total Spectra Processing
    # ----------------------------------------
    fig_total: go.Figure = go.Figure()
    fig_state: go.Figure = go.Figure()

    if state_mtime > 0.0:
        df_state: pl.DataFrame = pl.read_ipc(state_file)
        total_spectra_data = np.zeros(len(X_ENERGY), dtype=float)
        for row in df_state.iter_rows(named=True):
            state: str = str(row.get("state_label", "Unknown State"))
            spectra_data: list[float] = row.get("spectra", [])
            events: int = int(row.get("events", sum(spectra_data)))
            total_spectra_data += np.array(spectra_data)

            fig_state.add_trace(go.Scatter(x=X_ENERGY, y=spectra_data, mode="lines", name=f"{state} ({events:,} events)"))

        # Plot the accumulated total spectrum
        total_events = int(np.sum(total_spectra_data))
        fig_total.add_trace(
            go.Scatter(
                x=X_ENERGY,
                y=total_spectra_data,
                mode="lines",
                name=f"All Channels ({total_events:,} events)",
                fill="tozeroy",  # Optional: Adds a nice visual weight to the total plot
            )
        )

    if total_fig and "layout" in total_fig:
        fig_total.update_layout(**total_fig["layout"])
    else:
        fig_total.update_layout(title="Total Spectrum (all channels)", **DEFAULT_LAYOUT)
    if state_fig and "layout" in state_fig:
        fig_state.update_layout(**state_fig["layout"])
    else:
        fig_state.update_layout(title="State Spectra", **DEFAULT_LAYOUT)

    # ----------------------------------------
    # 2. Channel Spectra Processing
    # ----------------------------------------
    fig_chan: go.Figure = go.Figure()

    if chan_mtime > 0.0:
        df_chan: pl.DataFrame = pl.read_ipc(chan_file).sort("channel_number")
        for row in df_chan.iter_rows(named=True):
            channel_num: Any = row.get("channel_number", "Unknown")
            chan_spectra_data: list[float] = row.get("spectra", [])
            events2: int = int(row.get("events", sum(chan_spectra_data)))

            label_str = str(channel_num)
            trace_name = label_str if "Chan" in label_str else f"Chan {label_str}"

            fig_chan.add_trace(go.Scatter(x=X_ENERGY, y=chan_spectra_data, mode="lines", name=f"{trace_name} ({events2:,} events2)"))

    if chan_fig and "layout" in chan_fig:
        fig_chan.update_layout(**chan_fig["layout"])
    else:
        fig_chan.update_layout(title="Channel Spectra", **DEFAULT_LAYOUT)

    return fig_total, fig_state, fig_chan, current_server_version


if __name__ == "__main__":
    app.run(debug=True)
