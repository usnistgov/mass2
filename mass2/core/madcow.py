import argparse
import numpy as np
import polars as pl
import pyarrow as pa
from pyarrow import ipc
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from numpy.typing import NDArray
from typing import cast, BinaryIO
from watchdog.observers import Observer
from .massassin import FileModifiedHandler

from .massassin import load_expt_state_df, attach_tz_to_naive_column, ARROW_EOS_MARKER
from .misc import str2channum, chanfile_prefix


def add_expt_state(df: pl.DataFrame, df_estate: pl.DataFrame, time_col: str = "timestamp") -> pl.DataFrame:
    """Add experiment state column named "state_label" to a data frame

    Parameters
    ----------
    df : pl.DataFrame
        The data frame to be supplemented with state labels
    df_estate : pl.DataFrame
        A table of state labels and the times that they start
    time_col : str, optional
        The dataframe column name to be used for time-matching, by default "timestamp"

    Returns
    -------
    pl.DataFrame
        The updated `df` now with a "state_label" column
    """
    # 1. Add a temporary row index to remember the original order
    df = df.with_row_index("__original_order__")

    # 2. Sort both DataFrames by the timestamp (REQUIRED for join_asof)
    df_sorted = df.sort(time_col)
    df_sorted = attach_tz_to_naive_column(df_sorted, time_col, "UTC")
    df_estate_sorted = df_estate.sort(time_col)

    # 3. Perform the as-of join
    # strategy="backward" (the default) matches the last earlier or exact time
    joined = df_sorted.join_asof(df_estate_sorted, on=time_col, strategy="backward")

    # 4. Sort back to the original order and drop the temporary index
    return joined.sort("__original_order__").drop("__original_order__")


def raw_arrows_timezone(input_dir: Path, default_tz: str = "UTC") -> str:
    inputs_sortedP = list(input_dir.glob("*_chan*.parquet"))
    if len(inputs_sortedP) > 0:
        lf = pl.scan_parquet(inputs_sortedP[0])
        dtype = lf.collect_schema()["timestamp"]
        dtype = cast(pl.Datetime, dtype)
        return default_tz if dtype.time_zone is None else str(dtype.time_zone)

    inputs_sortedA = list(input_dir.glob("*_chan*.arrow"))
    if len(inputs_sortedA) > 0:
        lf = pl.scan_ipc(inputs_sortedA[0])
        dtype = lf.collect_schema()["timestamp"]
        dtype = cast(pl.Datetime, dtype)
        return default_tz if dtype.time_zone is None else str(dtype.time_zone)

    inputs_unsorted = list(input_dir.glob("*.arrows*"))
    if len(inputs_unsorted) > 0:
        with pa.ipc.open_stream(inputs_unsorted[0]) as reader:
            tz = reader.schema.field("timestamp").type.tz
            return default_tz if tz is None else tz
    raise OSError(f"found no valid '*_chan*.arrow' or '*.arrows*' files in {input_dir}")


@dataclass(frozen=False)
class MadCowDirectory:
    """Object to run MAD-COW on a single data directory"""

    input_dir: Path
    output_dir: Path
    expt_state_path: Path
    expt_state_df: pl.DataFrame
    Emin: float
    Emax: float
    Nbins: int
    state_spectra: dict[str, NDArray] = field(default_factory=dict)
    chan_spectra: dict[int, NDArray] = field(default_factory=dict)
    file_prefix: str | None = None

    @classmethod
    def open(cls, input_dir: str | Path, output_dir: str | Path, Emin: float, Emax: float, Nbins: int) -> "MadCowDirectory":
        """Create a new MadCowDirectory

        Parameters
        ----------
        input_dir : str | Path
            _description_
        output_dir : str | Path
            _description_
        Emin : float
            _description_
        Emax : float
            _description_
        Nbins : int
            _description_

        Returns
        -------
        MadCowDirectory
            _description_
        """
        input_dir = Path(input_dir)
        target_time_zone = raw_arrows_timezone(input_dir)

        state_files = list(input_dir.glob("*_experiment_state.txt"))
        parent_state_files = list(input_dir.parent.glob("*_experiment_state.txt"))
        assert len(state_files) + len(parent_state_files) > 0, f"found no experiment state file in {input_dir} or its parent"
        if len(state_files) == 0:
            state_files = parent_state_files
        assert len(state_files) == 1, f"found {len(state_files)} '*_experiment_state.txt' files in {input_dir}, want exactly 1"
        expt_state_df = load_expt_state_df(state_files[0], target_time_zone)
        md = cls(Path(input_dir), Path(output_dir), Path(state_files[0]), expt_state_df, Emin, Emax, Nbins)

        # Check that input exists and contains raw pulse files and an experiment_state.txt file
        md.validate_input()

        # Create and check output directory
        md.validate_output()

        return md

    def validate_input(self) -> None:
        """Ensure that the given data directory is a directory and contains at least one appropriate data file

        Parameters
        ----------
        input_dir : Path
            The data directory where raw pulse files live.
        """
        assert self.Emin < self.Emax
        assert self.Nbins > 1
        input_dir = self.input_dir
        assert input_dir.exists(), f"{input_dir=} does not exist"
        assert input_dir.is_dir(), f"{input_dir=} is not a directory"
        onechan_files = list(input_dir.glob("*_chan*.parquet"))
        if len(onechan_files) > 0:
            f = Path(onechan_files[0])
            prefix = chanfile_prefix(f)
            assert prefix, f"did not find pattern *_chan[digits] in file {f}"
            self.file_prefix = prefix
            return

        arrow_files = list(input_dir.glob("*[0-9].arrow*"))
        assert len(arrow_files) > 0, f"{input_dir=} contains no Arrows files"
        f = Path(arrow_files[0])
        prefix = chanfile_prefix(f, chantext="")
        assert prefix, f"did not find pattern *_chan[digits] in file {f}"
        self.file_prefix = prefix
        return

    def validate_output(self) -> None:
        """Ensure that the given output directory exists or can be created.

        Parameters
        ----------
        output_dir : Path
            The data directory where analyzed pulse results will be written.
        """
        output_dir = self.output_dir
        Path.mkdir(output_dir, mode=0o755, parents=True, exist_ok=True)
        assert output_dir.exists(), f"{output_dir=} could not be made"
        assert output_dir.is_dir(), f"{output_dir=} is not a directory"

    def process_singlechan(self, parquet_file: Path, channum: int, time_col: str = "timestamp") -> None:
        """Process the raw pulse data from a single channel with the given recipe

        Parameters
        ----------
        parquet_file : Path
            File path containing the analyzed pulse data in a Parquet file
        """
        print(f"Analzying single-channel {parquet_file.name}")
        states_lazy = self.expt_state_df.lazy().sort(time_col)
        df_joined = (
            pl.scan_parquet(parquet_file)
            # Remove data that fails cuts
            .filter(pl.col("good"))
            # Sort by the join key (timestamp), or tell Polars that they already ARE sorted. In this case, the latter
            .with_columns(pl.col(time_col).set_sorted())
            .join_asof(states_lazy, on=time_col, strategy="backward")
            .select(["energy1", "state_label"])
            .collect()
        )
        self.update_histograms(df_joined)

    def update_histograms(self, df_joined: pl.DataFrame) -> None:
        fixed_E_bins = np.linspace(self.Emin, self.Emax, 1 + self.Nbins)
        hist_by_channel = df_joined.group_by("channel_number").agg(pl.col("energy1").hist(fixed_E_bins).alias("hist"))
        hist_by_state = df_joined.group_by("state_label").agg(pl.col("energy1").hist(fixed_E_bins).alias("hist"))

        for row in hist_by_channel.iter_rows(named=True):
            channum = row["channel_number"]
            histogram = np.array(row["hist"])
            if channum in self.chan_spectra:
                self.chan_spectra[channum] += histogram
            else:
                self.chan_spectra[channum] = histogram

        for row in hist_by_state.iter_rows(named=True):
            state = row["state_label"]
            if not state:
                continue
            histogram = np.array(row["hist"])
            if state in self.state_spectra:
                self.state_spectra[state] += histogram
            else:
                self.state_spectra[state] = histogram

    def analyze_old_data(self) -> bool:
        """Analyze "old data", meaning data that has already been unshuffled into single-channel files.

        Returns
        -------
        bool
            Whether an old data set was found, and analyzed
        """
        per_chan_files = list(self.input_dir.glob("*_chan*.parquet"))
        if len(per_chan_files) == 0:
            return False

        # Analyze the spectra
        for parquet_file in per_chan_files:
            channum = str2channum(parquet_file)
            assert channum is not None, f"could not parse channel number from file {parquet_file=}"
            self.process_singlechan(parquet_file, channum)

        self.write_histograms()
        return True

    def write_histograms(self) -> None:
        dfs = pl.DataFrame(
            {
                "state_label": list(self.state_spectra.keys()),
                "Emin": self.Emin,
                "Emax": self.Emax,
                "spectra": list(self.state_spectra.values()),
            },
            schema={"state_label": pl.String, "Emin": pl.Float64, "Emax": pl.Float64, "spectra": pl.Array(pl.Int32, self.Nbins)},
        )
        dfs.write_ipc(self.output_dir / "state_spectra.arrow")
        dfc = pl.DataFrame(
            {
                "channel_number": list(self.chan_spectra.keys()),
                "Emin": self.Emin,
                "Emax": self.Emax,
                "spectra": list(self.chan_spectra.values()),
            },
            schema={"channel_number": pl.Int32, "Emin": pl.Float64, "Emax": pl.Float64, "spectra": pl.Array(pl.Int32, self.Nbins)},
        )
        dfc.write_ipc(self.output_dir / "channel_spectra.arrow")

    def process_one_allchan_file(self, ipc_file: Path, output: Path, time_col: str = "timestamp") -> None:
        """Process the raw pulse data from a single channel with the given recipe

        Parameters
        ----------
        ipc_file : Path
            File path containing the raw pulse data in an Arrow IPC feather file
        output : Path
            File path for writing the output dataframe, as Parquet.
        """
        print(f"Analzying all-channel {ipc_file.name}")
        states = self.expt_state_df.sort(time_col)
        df = pl.read_ipc_stream(ipc_file)
        df = attach_tz_to_naive_column(df, time_col, "UTC")

        df = (
            # Sort by the join key (timestamp), or tell Polars that they already ARE sorted. In this case, the former.
            df.filter(pl.col("good"))
            .select(["channel_number", time_col, "energy1"])
            .with_columns(pl.col(time_col).sort())
            .join_asof(states, on=time_col, strategy="backward")
            .drop(time_col)
        )
        self.update_histograms(df)

    def analyze_WAL_tail(self, wal_path: Path) -> bool:
        modified_event = threading.Event()
        event_handler = FileModifiedHandler(wal_path, modified_event)
        observer = Observer()
        observer.schedule(event_handler, path=str(wal_path.parent), recursive=False)
        observer.start()

        try:
            with open(wal_path, "rb") as fp:
                try:
                    reader = ipc.RecordBatchStreamReader(fp)
                    reader_schema = reader.schema
                except pa.ArrowInvalid:
                    print("File is too new, schema not written yet.")
                    return False

                # Pass the modified_event into the processing loop
                self._analyze_open_WAL(fp, reader_schema, modified_event, wal_path)
                return True
        finally:
            observer.stop()
            observer.join()

    def _analyze_open_WAL(self, fp: BinaryIO, reader_schema: pa.schema, modified_event: threading.Event, wal_path: Path) -> None:
        """Process incoming PyArrow batches from a Write-Ahead Log in real time."""

        last_good_position = fp.tell()
        # Sort states once for the entire file stream to use in asof joins
        states = self.expt_state_df.sort("timestamp")

        last_write_time = time.time()
        while True:
            # Peek to see if the 8-byte-long EOS (end-of-stream) marker is next.
            fp.seek(last_good_position)
            header = fp.read(8)

            if len(header) < 8:
                # Check if the file was renamed by the DAQ (stream finalized abruptly)
                if not wal_path.exists():
                    print("WAL file no longer exists. Assuming DAQ renamed it; stream finalized.")
                    break
                # Torn Write or EOF: Wait for DAQ to write more data
                modified_event.wait(timeout=1.0)
                modified_event.clear()
                continue

            if header == ARROW_EOS_MARKER:
                print("Received EOS marker. Stream finalized.")
                break

            fp.seek(last_good_position)

            try:
                msg = ipc.read_message(fp)
                batch = ipc.read_record_batch(msg, reader_schema)
                last_good_position = fp.tell()

                # Convert zero-copy to Polars
                df_batch = pl.DataFrame(batch)

                # Apply the same filtering and joining logic used in process_one_allchan_file
                df_batch = attach_tz_to_naive_column(df_batch, "timestamp", "UTC")
                df_joined = (
                    df_batch.filter(pl.col("good"))
                    .select(["channel_number", "timestamp", "energy1"])
                    .with_columns(pl.col("timestamp").sort())
                    .join_asof(states, on="timestamp", strategy="backward")
                    .drop("timestamp")
                )

                # Update the running histogram dictionaries
                self.update_histograms(df_joined)
                if time.time() - last_write_time > 1:
                    last_write_time = time.time()
                    self.write_histograms()

            except pa.ArrowInvalid:
                # Torn Write: Header is complete but payload hasn't flushed yet
                modified_event.wait(timeout=1.0)
                modified_event.clear()

    def run(self) -> None:
        """Run analysis recipe on live-streaming data, including a cold-start phase."""

        # TODO set up watchdog for expt state file, to re-generate the expt state dataframe.

        seqnum = 0
        FILE_POLL_TIME = 0.2  # wait this many seconds before checking whether the next file exists yet.
        MAX_WAIT_TIME = 10.0  # wait this many seconds for the next file to exist before giving up.
        while True:
            # Compute the filename for this sequence number, whether finalized or in progress
            finalized_path = self.input_dir / f"{self.file_prefix}_{seqnum:04d}.arrows"
            wal_path = self.input_dir / f"{self.file_prefix}_{seqnum:04d}.arrows_WAL"

            # ---------------------------------------------------------
            # CASE 1: COLD START (run recipe on a finalized file)
            # ---------------------------------------------------------
            if finalized_path.exists():
                print(f"[{seqnum:04d}] Analyzing finalized file: {finalized_path}")
                output_path = self.output_dir / finalized_path.name
                self.process_one_allchan_file(finalized_path, output_path)
                self.write_histograms()
                seqnum += 1
                continue  # Immediately jump to the next sequence number

            # ---------------------------------------------------------
            # CASE 2: LIVE TAILING (run recipe on a write-ahead log)
            # ---------------------------------------------------------
            if wal_path.exists():
                print(f"[{seqnum:04d}] Analyzing WAL       file: {wal_path}")
                try:
                    success = self.analyze_WAL_tail(wal_path)
                    if success:
                        self.write_histograms()
                        seqnum += 1
                except FileNotFoundError:
                    # EDGE CASE PROTECTION: The Go DAQ closed and renamed the file
                    # in the microsecond between our os.path.exists() and our open().
                    # We catch it, ignore it, and let the loop restart to catch it as Phase 1!
                    pass
                time.sleep(FILE_POLL_TIME)
                continue

            # ---------------------------------------------------------
            # CASE 3: WAIT FOR DATA (next seqnum doesn't exist yet)
            # ---------------------------------------------------------
            print(f"[{seqnum:04d}] Waiting for new DAQ file...")
            start_wait = time.time()
            found = False

            while time.time() - start_wait < MAX_WAIT_TIME:
                # Check if either the WAL or a finalized file popped into existence
                # TODO there might be a kind of flag to tell us there won't be a next file, so we can stop waiting.
                if wal_path.exists() or finalized_path.exists():
                    found = True
                    break

                # Simple polling is perfectly fine here. We are only checking directory entries,
                # which is an ultra-cheap OS operation, and it only happens between file rotations.
                time.sleep(FILE_POLL_TIME)

            if not found:
                print(f"[{seqnum:04d}] Timeout: No new file appeared within {MAX_WAIT_TIME} seconds. Shutting down.")
                print(f"Was seeking {wal_path=}")
                print(f"Was seeking {finalized_path=}")
                break


def main_madcow() -> None:
    description = """Microcalorimeter Analysis Display - Compilation Online Worker.
Compile MASS results to spectra, either a complete or a live data set. MAD-Dash will display them."""
    output_help = "write output to this directory, (default: $input_dir)"

    parser = argparse.ArgumentParser(description=description)
    # Using type=Path directly parses the string into a Path object
    parser.add_argument("input_dir", type=Path, help="the directory to watch for raw pulse data")
    parser.add_argument("output_dir", type=Path, nargs="?", default=None, help=output_help)
    # parser.add_argument("-d", "--delayed", action="store_true", help="input contains old data; no need to monitor for new")

    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = args.input_dir

    md = MadCowDirectory.open(args.input_dir, args.output_dir, Emin=0, Emax=1000.0, Nbins=4000)

    # First detect and process unshuffled (single-channel) data. Assume that if any are found, they are all that matters.
    if md.analyze_old_data():
        return

    md.run()


if __name__ == "__main__":
    main_madcow()
