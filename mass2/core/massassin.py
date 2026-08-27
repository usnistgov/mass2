import argparse
import glob
import pickle
import polars as pl
import pyarrow as pa
import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import mass2


def cold_start(recipes: dict[int, mass2.core.Recipe], input_dir: Path, output_dir: Path) -> None:
    pass


def run_recipe(recipe: mass2.core.Recipe, raw_df: pl.DataFrame) -> pl.DataFrame:
    outputs = [
        "channel_number",
        "good",
        "timestamp",
        "subframecount",
        "pretrig_mean",
        "5lagx",
        "5lagy",
        "energy",
    ]

    # 3. Process the recipe
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


def load_expt_state_df(expt_state_file: str, target_time_zone: str) -> pl.DataFrame:
    df = pl.read_csv(expt_state_file, new_columns=["unixnano", "state_label"])
    df_es = df.select(pl.from_epoch("unixnano", time_unit="ns").dt.cast_time_unit("us").alias("timestamp"))
    df_labels = df.select(pl.col("state_label").str.strip_chars()).cast(pl.Categorical)
    times = df_es["timestamp"]
    times = times.dt.convert_time_zone(target_time_zone)
    df_es = df_es.with_columns(timestamp=times)
    return df_es.with_columns(df_labels)


def add_expt_state(df: pl.DataFrame, df_estate: pl.DataFrame, time_col: str = "timestamp") -> pl.DataFrame:
    # 1. Add a temporary row index to remember the original order
    df = df.with_row_index("__original_order__")

    # 2. Sort both DataFrames by the timestamp (REQUIRED for join_asof)
    df_sorted = df.sort(time_col)
    df_estate_sorted = df_estate.sort(time_col)

    # 3. Perform the as-of join
    # strategy="backward" (the default) matches the last earlier or exact time
    joined = df_sorted.join_asof(df_estate_sorted, on=time_col, strategy="backward")

    # 4. Sort back to the original order and drop the temporary index
    return joined.sort("__original_order__").drop("__original_order__")


def raw_arrows_timezone(input_dir: Path) -> str:
    inputs_sorted = glob.glob(str(input_dir / "*_chan*.arrow"))
    inputs_unsorted = glob.glob(str(input_dir / "*.arrows*"))
    if len(inputs_sorted) > 0:
        lf = pl.scan_ipc(inputs_sorted[0])
        dtype = lf.collect_schema()["timestamp"]
        dtype = cast(pl.Datetime, dtype)
        return str(dtype.time_zone)
    if len(inputs_unsorted) > 0:
        with pa.ipc.open_stream(inputs_unsorted[0]) as reader:
            return reader.schema.field("timestamp").type.tz
    raise OSError(f"found no valid '*_chan*.arrow' or '*.arrows*' files in {input_dir}")


@dataclass(frozen=True)
class MassassinDirectory:
    recipes: dict[int, mass2.core.Recipe]
    recipe_file: Path
    input_dir: Path
    output_dir: Path
    expt_state_df: pl.DataFrame

    @classmethod
    def open(cls, recipe_file: str | Path, input_dir: str | Path, output_dir: str | Path) -> "MassassinDirectory":
        with open(recipe_file, "rb") as fp:
            recipes = pickle.load(fp)
        input_dir = Path(input_dir)
        target_time_zone = raw_arrows_timezone(input_dir)

        expt_state_df = pl.DataFrame()
        state_files = glob.glob(str(input_dir / "*_experiment_state.txt"))
        if len(state_files) > 0:
            expt_state_df = load_expt_state_df(state_files[0], target_time_zone)
        return cls(recipes, Path(recipe_file), Path(input_dir), Path(output_dir), expt_state_df)

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
        assert len(glob.glob(str(input_dir / "*.arrow*"))) > 0, f"{input_dir=} contains no Arrows files"

    def validate_output(self) -> None:
        """Ensure that the given output directory can be created, or exists

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

    def process_singlechan(self, recipe: mass2.core.Recipe, ipc_file: str, output: Path) -> None:
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
        print(f"Analzying {input.name}")
        df_in = pl.read_ipc(input, memory_map=True)
        df = run_recipe(recipe, df_in)
        df = add_expt_state(df, self.expt_state_df)
        df.write_parquet(output)

    def analyze_old_data(self) -> bool:
        """_summary_

        Returns
        -------
        bool
            Whether an old data set was found, and analyzed
        """
        per_chan_files = glob.glob(str(self.input_dir / "*_chan*.arrow"))
        if len(per_chan_files) == 0:
            return False

        for ipc_file in per_chan_files:
            name = Path(ipc_file).name
            output = self.output_dir / name
            channum = self.channum(name)
            assert channum >= 0, f"could not parse channel number from file {name=}"
            try:
                recipe = self.recipes[channum]
            except KeyError:
                pass
            self.process_singlechan(recipe, ipc_file, output)

        return True

    @staticmethod
    def channum(name: str) -> int:
        stem = Path(name).stem
        match = re.search(r".*chan(\d+)$", stem)
        if match:
            return int(match.group(1))
        return -1

    def run(self) -> None:
        # Check that input exists and contains raw pulse files
        self.validate_input()

        # Create and check output directory
        self.validate_output()

        if self.analyze_old_data():
            return

        # TODO: Create 0MQ subscriber

        # TODO: cold start
        # cold_start(recipes, self.input_dir, self.output_dir)

        # TODO: streaming phase
        # TODO: When streaming is done, shuffle by channel.


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
    md.run()


if __name__ == "__main__":
    main_massassin()
