import argparse
import pickle
import polars as pl
import pyarrow as pa
from pyarrow import ipc
import time
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast, BinaryIO

import mass2
from .misc import str2channum, chanfile_prefix


ARROW_EOS_MARKER = b"\xff\xff\xff\xff\x00\x00\x00\x00"  # End-of-stream marker (8 bytes long)


def run_recipe(recipe: mass2.core.Recipe, raw_df: pl.DataFrame) -> pl.DataFrame:
    """Run a Mass2 recipe on the given dataframe

    Parameters
    ----------
    recipe : mass2.core.Recipe
        The recipe to run
    raw_df : pl.DataFrame
        The data to process

    Returns
    -------
    pl.DataFrame
        Result of the recipe run on `raw_df`
    """
    outputs = [
        "good",
        "timestamp",
        "subframecount",
        "pretrig_mean",
        "5lagx",
        "5lagy",
        "energy1",
    ]
    if "channel_number" in raw_df.columns:
        outputs.append("channel_number")

    framer = mass2.misc.DataFramerPolars(raw_df["pulse"])
    df = recipe.calc_from_df(raw_df, framer)
    good = pl.lit(True)
    for step in recipe.steps[::-1]:
        try:
            good = step.good_expr
            break
        except AttributeError:
            pass
    return df.with_columns(good=good).select(outputs)


def attach_tz_to_naive_column(df: pl.DataFrame, time_col: str, new_tz: str) -> pl.DataFrame:
    """If a given column of a dataframe is a Datetime lacking a time zone, attach the given time zone by name.

    Parameters
    ----------
    df : pl.DataFrame
        Existing data frame with a timestamp column
    time_col : str
        Name of the timestamp column (must be of type pl.Datetime in the schema)
    new_tz : str
        The time zone to attach, if the named column lacks a time zone

    Returns
    -------
    pl.DataFrame
        The updated data frame.
    """
    time_type = df.schema[time_col]
    assert isinstance(time_type, pl.Datetime)
    if time_type.time_zone is None:
        df = df.with_columns(pl.col(time_col).dt.replace_time_zone(new_tz))
    return df


def load_expt_state_df(expt_state_file: Path, target_time_zone: str) -> pl.DataFrame:
    """Load the experiment state file as a small DataFrame

    Parameters
    ----------
    expt_state_file : str
        File name to load
    target_time_zone : str
        Convert the timestamp column to this time zone

    Returns
    -------
    pl.DataFrame
        _description_
    """
    df = pl.read_csv(expt_state_file, new_columns=["timestamp", "state_label"])
    df_es = df.select(pl.from_epoch("timestamp", time_unit="ns").dt.cast_time_unit("us"))
    df_es = attach_tz_to_naive_column(df_es, "timestamp", target_time_zone)
    df_labels = df.select(pl.col("state_label").str.strip_chars()).cast(pl.Categorical)
    times = df_es["timestamp"].dt.convert_time_zone(target_time_zone)
    return df_es.with_columns(df_labels, timestamp=times)


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
    inputs_sorted = list(input_dir.glob("*_chan*.arrow"))
    inputs_unsorted = list(input_dir.glob("*.arrows*"))
    if len(inputs_sorted) > 0:
        lf = pl.scan_ipc(inputs_sorted[0])
        dtype = lf.collect_schema()["timestamp"]
        dtype = cast(pl.Datetime, dtype)
        return default_tz if dtype.time_zone is None else str(dtype.time_zone)
    if len(inputs_unsorted) > 0:
        with pa.ipc.open_stream(inputs_unsorted[0]) as reader:
            tz = reader.schema.field("timestamp").type.tz
            return default_tz if tz is None else tz
    raise OSError(f"found no valid '*_chan*.arrow' or '*.arrows*' files in {input_dir}")


RECIPE_OUTPUTS = (
    "channel_number",
    "good",
    "timestamp",
    "subframecount",
    "pretrig_mean",
    "5lagx",
    "5lagy",
    "energy1",
)


class FileModifiedHandler(FileSystemEventHandler):
    """An event handler to set a threading event when a specified file is modified."""

    def __init__(self, target_file: Path, modified_event: threading.Event):
        """Initialize the event handler

        Parameters
        ----------
        target_file : Path
            File that will be watched
        modified_event : threading.Event
            The event to handle
        """
        self.target_file = target_file.resolve()
        self.modified_event = modified_event

    def on_modified(self, event: FileSystemEvent) -> None:
        # Trigger the event only if the modified file is the exact file we are watching
        if Path(str(event.src_path)).resolve() == self.target_file:
            self.modified_event.set()


@dataclass(frozen=False)
class MassassinDirectory:
    recipes: dict[int, mass2.core.Recipe]
    recipe_file: Path
    input_dir: Path
    output_dir: Path
    expt_state_path: Path
    expt_state_df: pl.DataFrame
    file_ids_complete: set[int] = field(default_factory=set)
    file_prefix: str | None = None

    @classmethod
    def open(cls, recipe_file: str | Path, input_dir: str | Path, output_dir: str | Path) -> "MassassinDirectory":
        with open(recipe_file, "rb") as fp:
            recipes = pickle.load(fp)
        input_dir = Path(input_dir)
        target_time_zone = raw_arrows_timezone(input_dir)

        state_files = list(input_dir.glob("*_experiment_state.txt"))
        assert len(state_files) > 0, f"found no experiment state file in {input_dir}"
        assert len(state_files) == 1, f"found {len(state_files)} '*_experiment_state.txt' files in {input_dir}, want exactly 1"
        expt_state_df = load_expt_state_df(state_files[0], target_time_zone)
        md = cls(recipes, Path(recipe_file), Path(input_dir), Path(output_dir), Path(state_files[0]), expt_state_df)

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
        input_dir = self.input_dir
        assert input_dir.exists(), f"{input_dir=} does not exist"
        assert input_dir.is_dir(), f"{input_dir=} is not a directory"
        onechan_files = list(input_dir.glob("*_chan*.arrow"))
        if len(onechan_files) > 0:
            f = onechan_files[0]
            prefix = chanfile_prefix(f)
            assert prefix, f"did not find pattern *_chan[digits] in file {f}"
            self.file_prefix = prefix
            return

        arrow_files = list(input_dir.glob("*.arrow*"))
        assert len(arrow_files) > 0, f"{input_dir=} contains no Arrows files"
        f = arrow_files[0]
        prefix = chanfile_prefix(f, chantext="")
        assert prefix, f"did not find pattern *_[digits] in file {f}"
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
        assert not output_dir.samefile(self.input_dir)
        assert output_dir.exists(), f"{output_dir=} could not be made"
        assert output_dir.is_dir(), f"{output_dir=} is not a directory"

    def process_singlechan(self, recipe: mass2.core.Recipe, ipc_file: Path, output: Path) -> None:
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
        print(f"Analzying single-channel {ipc_file.name}")
        df_in = pl.read_ipc(ipc_file, memory_map=True)
        df = run_recipe(recipe, df_in)
        df = add_expt_state(df, self.expt_state_df)
        df.write_parquet(output)

    def analyze_old_data(self) -> bool:
        """Analyze "old data", meaning data that has already been unshuffled into single-channel files.

        Returns
        -------
        bool
            Whether an old data set was found, and analyzed
        """
        per_chan_files = list(self.input_dir.glob("*_chan*.arrow"))
        if len(per_chan_files) == 0:
            return False

        for ipc_file in per_chan_files:
            channum = str2channum(ipc_file.name)
            assert channum is not None, f"could not parse channel number from file {ipc_file=}"
            name = ipc_file.stem + ".parquet"
            output = self.output_dir / name
            if channum not in self.recipes:
                print(f"   found no recipe to match chan {channum}, file '{ipc_file}'")
                continue

            recipe = self.recipes[channum]
            self.process_singlechan(recipe, ipc_file, output)

        return True

    def run_recipe(self, batch: pa.RecordBatch) -> pl.DataFrame:
        frames: list[pl.DataFrame] = []

        # 1. Convert the entire batch to Polars ONCE (zero-copy transfer)
        full_df = pl.DataFrame(batch)

        # 2. Partition the dataframe in a single O(N) pass.
        # partition_by() returns a list of DataFrames, one for each unique channel.
        for raw_df in full_df.partition_by("channel_number", maintain_order=False, include_key=True):
            # Grab the channel number from the first row of this chunk
            cnum = raw_df["channel_number"][0]
            if cnum not in self.recipes:
                continue

            # 3. Process the recipe
            framer = mass2.misc.DataFramerPolars(raw_df["pulse"])
            recipe = self.recipes[cnum]
            df = recipe.calc_from_df(raw_df, framer)
            good = pl.lit(True)
            for step in recipe.steps[::-1]:
                try:
                    good = step.good_expr
                    break
                except AttributeError:
                    pass
            df = df.with_columns(good=good).select(RECIPE_OUTPUTS)
            frames.append(df)

        # 4. Concat all processed frames
        return pl.concat(frames)

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
        input = Path(ipc_file)
        print(f"Analzying all-channel {input.name}")
        df_in = pl.read_ipc_stream(input)
        df = self.run_recipe(df_in)
        df.write_ipc_stream(output)

    def analyze_WAL_tail(self, wal_path: Path, output_path: Path) -> None:
        # Set up the threading event and Watchdog observer
        modified_event = threading.Event()
        event_handler = FileModifiedHandler(wal_path, modified_event)
        observer = Observer()
        # Watch the parent directory, since Watchdog monitors directories, not files
        observer.schedule(event_handler, path=str(wal_path.parent), recursive=False)
        observer.start()

        try:
            with open(wal_path, "rb") as fp:
                try:
                    reader = ipc.RecordBatchStreamReader(fp)
                    reader_schema = reader.schema
                except pa.ArrowInvalid:
                    print("File is too new, schema not written yet.")
                    return

                with open(output_path, "wb") as outp:
                    # Pass work off to a new method, simply to unindent by 2 levels.
                    self._analyze_open_WAL(fp, outp, reader_schema, modified_event)
        finally:
            # Ensure the background thread is cleaned up when the file is finalized
            observer.stop()
            observer.join()

    def _analyze_open_WAL(self, fp: BinaryIO, outp: BinaryIO, reader_schema: pa.schema, modified_event: threading.Event) -> None:
        last_good_position = fp.tell()
        writer: ipc.RecordBatchStreamWriter | None = None

        while True:
            fp.seek(last_good_position)

            # Peek to see if the 8-byte-long EOS (end-of-stream) marker is next.
            header = fp.read(8)

            if len(header) < 8:
                # Physical EOF (len=0): DAQ hasn't written the next batch or EOF yet, or
                # Torn Write (0<len<8): DAQ is not finished writing the batch or EOF.
                # Wait for watchdog to signal new data, then clear the flag
                # Here and below, the 1-second timeout is a failsafe in case watchdog misses an event.
                modified_event.wait(timeout=1.0)
                modified_event.clear()
                continue

            if header == ARROW_EOS_MARKER:
                # FOUND THE EOS MARKER! The DAQ closed the file cleanly.
                print("Received EOS marker. Stream finalized.")
                break

            # If the next 8 bytes are not the EOS marker, it's a real batch (either complete or partial).
            # Rewind the pointer so PyArrow can parse as a normal batch.
            fp.seek(last_good_position)

            try:
                # Let PyArrow read the full message, parse it, and update the bookmark
                msg = ipc.read_message(fp)
                batch = ipc.read_record_batch(msg, reader_schema)
                last_good_position = fp.tell()

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
                # Wait for payload to finish flushing to disk
                modified_event.wait(timeout=1.0)
                modified_event.clear()

    def run(self) -> None:
        """Run analysis recipe on live-streaming data, including a cold-start phase."""

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
                    # in the moment between our os.path.exists() and our open().
                    # We catch it, ignore it, and let the loop restart to find it in Phase 1!
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


def main_massassin() -> None:
    description = "Run a recipe on raw data, either a complete or a live data set"
    output_help = "write output to this directory, (default: $input_dir/mass)"

    parser = argparse.ArgumentParser(description=description)
    # Using type=Path directly parses the string into a Path object
    parser.add_argument("recipe_file", type=Path, help="the recipe file (saved by Mass2, generally as a *.pkl)")
    parser.add_argument("input_dir", type=Path, help="the directory to watch for raw pulse data")
    parser.add_argument("output_dir", type=Path, nargs="?", default=None, help=output_help)
    # parser.add_argument("-d", "--delayed", action="store_true", help="input contains old data; no need to monitor for new")

    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = args.input_dir / "mass"

    md = MassassinDirectory.open(args.recipe_file, args.input_dir, args.output_dir)

    # First detect and process unshuffled (single-channel) data. Assume that if any are found, they are all that matters.
    if md.analyze_old_data():
        return

    md.run()


if __name__ == "__main__":
    main_massassin()
