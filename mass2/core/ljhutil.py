import glob
import os
import re
import struct
import numpy as np
from typing import Union, Optional, BinaryIO
from collections.abc import Iterator
import pathlib
from packaging.version import Version

from .ljhfiles import LJHFile

__all__ = ["find_ljh_files", "ljh_truncate"]

# functions for finding ljh files and opening them as Channels


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


def find_ljh_files(folder: str, ext: str = ".ljh", search_subdirectories: bool = False) -> list[str]:
    """
    Finds all .ljh files in the given folder and its subfolders.

    Args:
    - folder (str): The root directory to start the search from.

    Returns:
    - list[str]: A list of paths to .ljh files.
    """
    ljh_files = []
    if search_subdirectories:
        pathgen = os.walk(folder)
    else:
        pathgen = zip([folder], [[""]], [os.listdir(folder)])
    for dirpath, _, filenames in pathgen:
        for filename in filenames:
            if filename.endswith(ext):
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
    match = re.search(r"_chan(\d+)\.ljh$", file_path)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"File path does not match expected pattern: {file_path}")


def match_files_by_channel(folder1: str, folder2: str, limit=None, exclude_ch_nums=[]) -> list[Iterator[tuple[str, str]]]:
    """
    Matches .ljh files from two folders by channel number.

    Args:
    - folder1 (str): The first root directory.
    - folder2 (str): The second root directory.

    Returns:
    - list[Iterator[tuple[str, str]]]: A list of iterators, each containing pairs of paths with matching channel numbers.
    """
    files1 = find_ljh_files(folder1)
    files2 = find_ljh_files(folder2)
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
        if channel in files2_by_channel.keys() and channel not in exclude_ch_nums:
            matching_pairs.append((files1_by_channel[channel], files2_by_channel[channel]))
    # print(f"in match_files_by_channel found {len(matching_pairs)} channel pairs, {limit=}")
    matching_pairs_limited = matching_pairs[:limit]
    # print(f"in match_files_by_channel found {len(matching_pairs)=} after limit of {limit=}")
    return matching_pairs_limited


def experiment_state_path_from_ljh_path(
    ljh_path: Union[str, pathlib.Path],
) -> pathlib.Path:
    ljh_path = pathlib.Path(ljh_path)  # Convert to Path if it's a string
    base_name = ljh_path.name.split("_chan")[0]
    new_file_name = f"{base_name}_experiment_state.txt"
    return ljh_path.parent / new_file_name


def external_trigger_bin_path_from_ljh_path(
    ljh_path: Union[str, pathlib.Path],
) -> pathlib.Path:
    ljh_path = pathlib.Path(ljh_path)  # Convert to Path if it's a string
    base_name = ljh_path.name.split("_chan")[0]
    new_file_name = f"{base_name}_external_trigger.bin"
    return ljh_path.parent / new_file_name


def ljh_sort_filenames_numerically(fnames: list[str], inclusion_list: Optional[list[int]] = None) -> list[str]:
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
    subframecount, timestamp_usec, trace = src.read_trace_with_timing(i)
    prefix = struct.pack("<Q", int(subframecount))
    dest.write(prefix)
    prefix = struct.pack("<Q", int(timestamp_usec))
    dest.write(prefix)
    trace.tofile(dest, sep="")


def ljh_append_traces(src_name: str, dest_name: str, pulses: Optional[range] = None) -> None:
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


def ljh_truncate(input_filename: str, output_filename: str, n_pulses: Optional[int] = None, timestamp: Optional[float] = None) -> None:
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
