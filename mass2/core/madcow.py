import argparse
import numpy as np
import polars as pl
import pyarrow as pa
from pyarrow import ipc
import time
from dataclasses import dataclass, field
from pathlib import Path
from numpy.typing import NDArray
from typing import cast, BinaryIO

from .massassin import load_expt_state_df, attach_tz_to_naive_column
from .misc import str2channum, chanfile_prefix


def add_expt_state(df: pl.DataFrame, df_estate: pl.DataFrame, time_col: str = "timestamp") -> pl.DataFrame:
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

        arrow_files = list(input_dir.glob("*.arrow*"))
        assert len(arrow_files) > 0, f"{input_dir=} contains no Arrows files"
        f = Path(arrow_files[0])
        prefix = chanfile_prefix(f)
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
        states_lazy = self.expt_state_df.lazy().sort(time_col)
        print(f"Analzying single-channel {parquet_file.name}")
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

        category_dfs = df_joined.partition_by("state_label", as_dict=True)
        all_states_hist = np.zeros(self.Nbins, dtype=int)

        for category_names, sub_df in category_dfs.items():
            category_name = category_names[0]
            if not category_name:
                continue
            assert isinstance(category_name, str), f"{category_name=}"
            if category_name not in self.state_spectra:
                self.state_spectra[category_name] = np.zeros(self.Nbins, dtype=int)
            data_array = sub_df.get_column("energy1").to_numpy()
            contents, _ = np.histogram(data_array, self.Nbins, (self.Emin, self.Emax))
            self.state_spectra[category_name] += contents
            all_states_hist += contents
        self.chan_spectra[channum] = all_states_hist

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
            assert channum, f"could not parse channel number from file {parquet_file=}"
            self.process_singlechan(parquet_file, channum)

        print("Yo!")
        print(self.state_spectra.keys())

        dfs = pl.DataFrame(
            {
                "state_label": list(self.state_spectra.keys()),
                "spectra": list(self.state_spectra.values()),
            },
            schema={"state_label": pl.String, "spectra": pl.Array(pl.Int32, self.Nbins)},
        )
        dfs.write_ipc(self.output_dir / "state_spectra.arrow")
        dfc = pl.DataFrame(
            {
                "channel_number": list(self.chan_spectra.keys()),
                "spectra": list(self.chan_spectra.values()),
            },
            schema={"channel_number": pl.Int32, "spectra": pl.Array(pl.Int32, self.Nbins)},
        )
        dfc.write_ipc(self.output_dir / "channel_spectra.arrow")
        return True

    def process_one_allchan_file(self, ipc_file: str | Path, output: Path) -> None:
        """Process the raw pulse data from a single channel with the given recipe

        Parameters
        ----------
        recipe : mass2.core.Recipe
            The recipe to run on the raw data
        ipc_file : str
            File path containing the raw pulse data in an Arrow IPC feather file
        output : Path
            File path for writing the output dataframe, as Parquet.
        """
        raise NotImplementedError
        input = Path(ipc_file)
        print(f"Analzying all-channel {input.name}")
        df_in = pl.read_ipc_stream(input)
        df = self.run_recipe(df_in)
        df.write_ipc_stream(output)

    def analyze_WAL_tail(self, wal_path: Path, output_path: Path) -> None:
        with open(wal_path, "rb") as fp:
            try:
                reader = ipc.RecordBatchStreamReader(fp)
                reader_schema = reader.schema
            except pa.ArrowInvalid:
                print("File is too new, schema not written yet.")
                return

            with open(output_path, "wb") as outp:
                # Pass work off to a new method, simply to unindent by 2 levels.
                self._analyze_open_WAL(fp, outp, reader_schema)

    def _analyze_open_WAL(self, fp: BinaryIO, outp: BinaryIO, reader_schema: pa.schema) -> None:
        """_summary_

        Parameters
        ----------
        fp : BinaryIO
            _description_
        outp : BinaryIO
            _description_
        reader_schema : pa.schema
            _description_

        Raises
        ------
        NotImplementedError
            _description_
        """
        FILE_POLL_TIME = 0.2  # wait this many seconds before checking whether the next file exists yet.
        last_good_position = fp.tell()
        writer: ipc.RecordBatchStreamWriter | None = None

        while True:
            fp.seek(last_good_position)

            # Peek to see if the 8-byte-long EOS (end-of-stream) marker is next.
            header = fp.read(8)

            if len(header) < 8:
                # Physical EOF (len=0): DAQ hasn't written the next batch or EOF yet, or
                # Torn Write (0<len<8): DAQ is not finished writing the batch or EOF.
                time.sleep(FILE_POLL_TIME)
                continue

            if header == b"\xff\xff\xff\xff\x00\x00\x00\x00":
                # FOUND THE EOS MARKER! The DAQ closed the file cleanly.
                print("Received EOS marker. Stream finalized.")
                break

            # If it is not the EOS marker, it's a real batch (either complete or partial).
            # Rewind the pointer exactly 8 bytes so PyArrow can parse it normally.
            fp.seek(last_good_position)

            try:
                # Let PyArrow read the full message, parse it, and update the bookmark
                msg = ipc.read_message(fp)
                batch = ipc.read_record_batch(msg, reader_schema)
                last_good_position = fp.tell()

                raise NotImplementedError

                df = self.run_recipe(batch).to_arrow()
                if len(df) == 0:
                    continue
                if not writer:
                    writer = ipc.new_stream(outp, df.schema)
                writer.write_table(df)
                outp.flush()  # force new data out of Python into OS page cache

            except pa.ArrowInvalid:
                # Torn Write: The header was complete, but the payload data
                # hasn't finished flushing to the disk yet.
                time.sleep(FILE_POLL_TIME)

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
                seqnum += 1
                continue  # Immediately jump to the next sequence number

            # ---------------------------------------------------------
            # CASE 2: LIVE TAILING (run recipe on a write-ahead log)
            # ---------------------------------------------------------
            if wal_path.exists():
                print(f"[{seqnum:04d}] Analyzing WAL       file: {wal_path}")
                try:
                    output_path = self.output_dir / wal_path.name
                    self.analyze_WAL_tail(wal_path, output_path)
                    seqnum += 1
                except FileNotFoundError:
                    # EDGE CASE PROTECTION: The Go DAQ closed and renamed the file
                    # in the microsecond between our os.path.exists() and our open().
                    # We catch it, ignore it, and let the loop restart to catch it as Phase 1!
                    pass
                continue

            # ---------------------------------------------------------
            # CASE 3: WAIT FOR DATA (next seqnum doesn't exist yet)
            # ---------------------------------------------------------
            print(f"[{seqnum:04d}] Waiting for new DAQ file...")
            # print(f"   {finalized_path}")
            # print(f"   {wal_path}")
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
