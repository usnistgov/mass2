"""
Utility functions for handling and finding LJH files, and opening them as Channel or Channels objects.
"""

import argparse
import glob
import os
import shutil
import re
import struct
import numpy as np
from typing import BinaryIO
import pathlib
from packaging.version import Version

from .ljhfiles import LJHFile

__all__ = ["find_ljh_files", "ljh_truncate"]


def find_folders_with_extension(root_path: str, extensions: list[str]) -> list[str]:
    """
    Finds all folders within the root_path that contain at least one file with the given extension.

    Args:
    - root_path (str): The root directory to start the search from.
    - extension (str): The file extension to search for (e.g., '.txt').

    Returns:
    - list[str]: A list of paths to directories containing at least one file with the given extension.
    """
    matching_folders = set()

    # Walk through the directory tree
    for dirpath, _, filenames in os.walk(root_path):
        # Check if any file in the current directory has the given extension
        for filename in filenames:
            for extension in extensions:
                if filename.endswith(extension):
                    matching_folders.add(dirpath)
                    break  # No need to check further, move to the next directory

    return list(matching_folders)


def find_ljh_files(
    folder: str | pathlib.Path,
    ext: str = ".ljh",
    search_subdirectories: bool = False,
    exclude_ch_nums: list[int] = [],
    include_ch_nums: list[int] | None = None,
) -> list[str]:
    """Finds all files of a specific file extension in the given folder and (optionally) its subfolders.

    An optional list of channel numbers can be excluded from the results. Also optionally, the results
    can be restricted only to a specific list of channel numbers.

    Parameters
    ----------
    folder : str | pathlib.Path
        Folder to search for data files
    ext : str, optional
        The filename extension to search for, by default ".ljh"
    search_subdirectories : bool, optional
        Whether to search the subdirectories of `folder` recursively, by default False
    exclude_ch_nums : list[int], optional
        List of channel numbers to exclude from the results, by default []
    include_ch_nums : list[int] | None, optional
        If not None, then a list of channel # such that results are excluded if they don't appear in the list, by default None

    Returns
    -------
    list[str]
        A list of paths to .ljh files.

    Raises
    ------
    ValueError
        When the `include_ch_nums` list exists and contains one or more channels also in `exclude_ch_nums`.
    """
    if include_ch_nums is not None:
        overlap = set(include_ch_nums).intersection(exclude_ch_nums)
        if len(overlap) > 0:
            raise ValueError(f"exclude and include lists should not overlap, but both include channels {overlap}")

    folder = str(folder)
    ljh_files = []
    if search_subdirectories:
        pathgen = os.walk(folder)
    else:
        pathgen = zip([folder], [[""]], [os.listdir(folder)])
    for dirpath, _, filenames in pathgen:
        for filename in filenames:
            if filename.endswith(ext):
                ch_num = extract_channel_number(filename)
                if ch_num in exclude_ch_nums:
                    continue
                if include_ch_nums is None or (ch_num in include_ch_nums):
                    ljh_files.append(os.path.join(dirpath, filename))
    return ljh_files


def extract_channel_number(file_path: str) -> int:
    """
    Extracts the channel number from the .ljh file name.

    Args:
    - file_path (str): The path to the .ljh file.

    Returns:
    - int: The channel number.
    """
    match = re.search(r"_chan(\d+)\..*$", file_path)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"File path does not match expected pattern: {file_path}")


def match_files_by_channel(
    folder1: str, folder2: str, limit: int | None = None, exclude_ch_nums: list[int] = [], include_ch_nums: list[int] | None = None
) -> list[tuple[str, str]]:
    """
    Matches .ljh files from two folders by channel number.

    Args:
    - folder1 (str): The first root directory.
    - folder2 (str): The second root directory.

    Returns:
    - list[Iterator[tuple[str, str]]]: A list of iterators, each containing pairs of paths with matching channel numbers.

    Raises
    ------
    ValueError
        When the `include_ch_nums` list exists and contains one or more channels also in `exclude_ch_nums`.
    """
    files1 = find_ljh_files(folder1, exclude_ch_nums=exclude_ch_nums, include_ch_nums=include_ch_nums)
    files2 = find_ljh_files(folder2, exclude_ch_nums=exclude_ch_nums, include_ch_nums=include_ch_nums)
    # print(f"in folder {folder1} found {len(files1)} files")
    # print(f"in folder {folder2} found {len(files2)} files")

    def collect_to_dict_error_on_repeat_channel(files: list[str]) -> dict:
        """
        Collects files into a dictionary by channel number, raising an error if a channel number is repeated.
        """
        files_by_channel: dict[int, str] = {}
        for file in files:
            channel = extract_channel_number(file)
            if channel in files_by_channel.keys():
                existing_file = files_by_channel[channel]
                raise ValueError(f"Duplicate channel number found: {channel} in file {file} and already in {existing_file}")
            files_by_channel[channel] = file
        return files_by_channel

    # we could have repeat channels even in the same folder, so we should error on that
    files1_by_channel = collect_to_dict_error_on_repeat_channel(files1)
    files2_by_channel = collect_to_dict_error_on_repeat_channel(files2)

    matching_pairs = []
    for channel in sorted(files1_by_channel.keys()):
        if channel in files2_by_channel.keys():
            matching_pairs.append((files1_by_channel[channel], files2_by_channel[channel]))
    if limit is not None:
        matching_pairs = matching_pairs[:limit]
    return matching_pairs


def experiment_state_path_from_ljh_path(
    ljh_path: str | pathlib.Path,
) -> pathlib.Path:
    """Find the experiment_state.txt file in the directory of the given ljh file."""
    ljh_path = pathlib.Path(ljh_path)  # Convert to Path if it's a string
    base_name = ljh_path.name.split("_chan")[0]
    new_file_name = f"{base_name}_experiment_state.txt"
    return ljh_path.parent / new_file_name


def external_trigger_bin_path_from_ljh_path(
    ljh_path: str | pathlib.Path,
) -> pathlib.Path:
    """Find the external_trigger.bin file in the directory of the given ljh file."""
    ljh_path = pathlib.Path(ljh_path)  # Convert to Path if it's a string
    base_name = ljh_path.name.split("_chan")[0]
    new_file_name = f"{base_name}_external_trigger.bin"
    return ljh_path.parent / new_file_name


def ljh_sort_filenames_numerically(fnames: list[str], inclusion_list: list[int] | None = None) -> list[str]:
    """Sort filenames of the form '*_chanXXX.*', according to the numerical value of channel number XXX.

    Filenames are first sorted by the usual string comparisons, then by channel number. In this way,
    the standard sort is applied to all files with the same channel number.

    :param fnames: A sequence of filenames of the form '*_chan*.*'
    :type fnames: list of str
    :param inclusion_list: If not None, a container with channel numbers. All files
        whose channel numbers are not on this list will be omitted from the
        output, defaults to None
    :type inclusion_list: sequence of int, optional
    :return: A list containg the same filenames, sorted according to the numerical value of channel number.
    :rtype: list
    """
    if fnames is None or len(fnames) == 0:
        return []

    if inclusion_list is not None:
        fnames = list(filter(lambda n: extract_channel_number(n) in inclusion_list, fnames))

    # Sort the results first by raw filename, then sort numerically by LJH channel number.
    # Because string sort and the builtin `sorted` are both stable, we ensure that the first
    # sort is used to break ties in channel number.
    fnames.sort()
    return sorted(fnames, key=extract_channel_number)


def filename_glob_expand(pattern: str) -> list[str]:
    """Return the result of glob-expansion on the input pattern.

    :param pattern: Aglob pattern and return the glob-result as a list.
    :type pattern: str
    :return: filenames; the result is sorted first by str.sort, then by ljh_sort_filenames_numerically()
    :rtype: list
    """
    result = glob.glob(pattern)
    return ljh_sort_filenames_numerically(result)


def helper_write_pulse(dest: BinaryIO, src: LJHFile, i: int) -> None:
    """Write a single pulse from one LJHFile to another open file."""
    subframecount, timestamp_usec, trace = src.read_trace_with_timing(i)
    prefix = struct.pack("<Q", int(subframecount))
    dest.write(prefix)
    prefix = struct.pack("<Q", int(timestamp_usec))
    dest.write(prefix)
    trace.tofile(dest, sep="")


def ljh_append_traces(src_name: str, dest_name: str, pulses: range | None = None) -> None:
    """Append traces from one LJH file onto another. The destination file is
    assumed to be version 2.2.0.

    Can be used to grab specific traces from some other ljh file, and append them onto an existing ljh file.

    Args:
        src_name: the name of the source file
        dest_name: the name of the destination file
        pulses: indices of the pulses to copy (default: None, meaning copy all)
    """

    src = LJHFile.open(src_name)
    if pulses is None:
        pulses = range(src.npulses)
    with open(dest_name, "ab") as dest_fp:
        for i in pulses:
            helper_write_pulse(dest_fp, src, i)


def ljh_truncate(input_filename: str, output_filename: str, n_pulses: int | None = None, timestamp: float | None = None) -> None:
    """Truncate an LJH file.

    Writes a new copy of an LJH file, with the same header but fewer raw data pulses.

    Arguments:
    input_filename  -- name of file to truncate
    output_filename -- filename for truncated file
    n_pulses        -- truncate to include only this many pulses (default None)
    timestamp       -- truncate to include only pulses with timestamp earlier
                       than this number (default None)

    Exactly one of n_pulses and timestamp must be specified.
    """

    if (n_pulses is None and timestamp is None) or (n_pulses is not None and timestamp is not None):
        msg = "Must specify exactly one of n_pulses, timestamp."
        msg += f" Values were {str(n_pulses)}, {str(timestamp)}"
        raise Exception(msg)

    # Check for file problems, then open the input and output LJH files.
    if os.path.exists(output_filename):
        if os.path.samefile(input_filename, output_filename):
            msg = f"Input '{input_filename}' and output '{output_filename}' are the same file, which is not allowed"
            raise ValueError(msg)

    infile = LJHFile.open(input_filename)
    if infile.ljh_version < Version("2.2.0"):
        raise Exception(f"Don't know how to truncate this LJH version [{infile.ljh_version}]")

    with open(output_filename, "wb") as outfile:
        # write the header as a single string.
        for k, v in infile.header.items():
            outfile.write(bytes(f"{k}: {v}\n", encoding="utf-8"))
        outfile.write(b"#End of Header\n")

        # Write pulses.
        if n_pulses is None:
            n_pulses = infile.npulses
        for i in range(n_pulses):
            if timestamp is not None and infile.datatimes_float[i] > timestamp:
                break
            prefix = struct.pack("<Q", np.uint64(infile.subframecount[i]))
            outfile.write(prefix)
            prefix = struct.pack("<Q", np.uint64(infile.datatimes_raw[i]))
            outfile.write(prefix)
            trace = infile.read_trace(i)
            trace.tofile(outfile, sep="")


def main_ljh_truncate() -> None:
    """
    A convenience script to truncate all LJH files that match a pattern, writing a new LJH file for each
    that contains only the first N pulse records.
    """
    parser = argparse.ArgumentParser(description="Truncate a set of LJH files")
    parser.add_argument("pattern", type=str, help="basename of files to process, e.g. 20171116_152922")
    parser.add_argument("out", type=str, help="string to append to basename when creating output filename")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--npulses", type=int, help="Number of pulses to keep")
    group.add_argument("--timestamp", type=float, help="Keep only pulses before this timestamp")
    args = parser.parse_args()

    pattern = f"{args.pattern}_chan*.ljh"

    filenames = filename_glob_expand(pattern)
    for in_fname in filenames:
        matches = re.search(r"chan(\d+)\.ljh", in_fname)
        if matches:
            ch = matches.groups()[0]
            out_fname = f"{args.pattern}_{args.out}_chan{ch}.ljh"
            ljh_truncate(in_fname, out_fname, n_pulses=args.npulses, timestamp=args.timestamp)


def ljh_merge(out_path: str, filenames: list[str], overwrite: bool) -> None:
    """Merge a set of LJH files to a single output file."""
    if not overwrite and os.path.isfile(out_path):
        raise OSError(f"To overwrite destination {out_path}, use the --force flag")
    shutil.copy(filenames[0], out_path)
    f = LJHFile.open(out_path)
    channum = f.channum
    print(f"Combining {len(filenames)} LJH files from channel {channum}")
    print(f"<-- {filenames[0]}")

    for in_fname in filenames[1:]:
        f = LJHFile.open(in_fname)
        if f.channum != channum:
            raise RuntimeError(f"file '{in_fname}' channel={f.channum}, but want {channum}")
        print(f"<-- {in_fname}")
        ljh_append_traces(in_fname, out_path)

    size = os.stat(out_path).st_size
    print(f"--> {out_path}    size: {size} bytes.\n")


def main_ljh_merge() -> None:
    """
    Merge all LJH files that match a pattern to a single output file.

    The idea is that all such files come from a single TES and could have been
    (but were not) written as a single continuous file.

    The pattern should be of the form "blah_blah_*_chan1.ljh" or something.
    The output will then be "merged_chan1.ljh" in the directory of the first file found
    (or alter the directory with the --outdir argument). It is not (currently) possible to
    merge data from LJH files that represent channels with different numbers.
    """
    parser = argparse.ArgumentParser(
        description="Merge a set of LJH files",
        epilog="Beware! Python glob does not perform brace-expansion, so braces must be expanded by the shell.",
    )
    parser.add_argument(
        "patterns", type=str, nargs="+", help='glob pattern of files to process, e.g. "20171116_*_chan1.ljh" (suggest double quotes)'
    )

    parser.add_argument(
        "-d",
        "--outdir",
        type=str,
        default="",
        help="directory to place output file (default: same as directory of first file to be merged",
    )
    # TODO: add way to control the output _filename_
    parser.add_argument("-F", "--force", action="store_true", help="force overwrite of existing target? (default: False)")
    parser.add_argument("-v", "--verbose", action="store_true", help="list files found before merging (default: False)")
    parser.add_argument("-n", "--dry-run", action="store_true", help="list files found, then quit without merging (default: False)")

    args = parser.parse_args()

    filenames: list[str] = []
    for pattern in args.patterns:
        filenames.extend(filename_glob_expand(pattern))
    assert len(filenames) > 0
    if args.verbose or args.dry_run:
        print(f"Will expand the following {len(filenames)} files:")
        for f in filenames:
            print("  - ", f)
        if args.dry_run:
            return

    ljh = LJHFile.open(filenames[0])
    channum = ljh.channum

    out_dir = args.outdir
    if not out_dir:
        out_dir = os.path.split(filenames[0])[0]
    out_path = os.path.join(out_dir, f"merged_chan{channum}.ljh")

    overwrite: bool = args.force
    ljh_merge(out_path, filenames, overwrite=overwrite)
