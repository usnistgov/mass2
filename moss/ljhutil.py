import os
import re
from typing import List, Iterator, Tuple, Union
import pathlib

# functions for finding ljh files and opening them as Channels


def find_folders_with_extension(root_path: str, extensions: List[str]) -> List[str]:
    """
    Finds all folders within the root_path that contain at least one file with the given extension.

    Args:
    - root_path (str): The root directory to start the search from.
    - extension (str): The file extension to search for (e.g., '.txt').

    Returns:
    - List[str]: A list of paths to directories containing at least one file with the given extension.
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


def find_ljh_files(folder: str, ext: str = ".ljh",
                   search_subdirectories: bool = False) -> List[str]:
    """
    Finds all .ljh files in the given folder and its subfolders.

    Args:
    - folder (str): The root directory to start the search from.

    Returns:
    - List[str]: A list of paths to .ljh files.
    """
    ljh_files = []
    if search_subdirectories:
        pathgen = os.walk(folder)
    else:
        pathgen = zip([folder], [None], [os.listdir(folder)])
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
    match = re.search(r'_chan(\d+)\.ljh$', file_path)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"File path does not match expected pattern: {file_path}")


def match_files_by_channel(folder1: str, folder2: str, limit=None) -> List[Iterator[Tuple[str, str]]]:
    """
    Matches .ljh files from two folders by channel number.

    Args:
    - folder1 (str): The first root directory.
    - folder2 (str): The second root directory.

    Returns:
    - List[Iterator[Tuple[str, str]]]: A list of iterators, each containing pairs of paths with matching channel numbers.
    """
    files1 = find_ljh_files(folder1)
    files2 = find_ljh_files(folder2)
    # print(f"in folder {folder1} found {len(files1)} files")
    # print(f"in folder {folder2} found {len(files2)} files")

    def collect_to_dict_error_on_repeat_channel(files: List[str]) -> dict:
        """
        Collects files into a dictionary by channel number, raising an error if a channel number is repeated.
        """
        files_by_channel = {}
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
    # print(f"in match_files_by_channel found {len(matching_pairs)} channel pairs, {limit=}")
    matching_pairs_limited = matching_pairs[:limit]
    # print(f"in match_files_by_channel found {len(matching_pairs)=} after limit of {limit=}")
    return matching_pairs_limited


def experiment_state_path_from_ljh_path(ljh_path: Union[str, pathlib.Path]) -> pathlib.Path:
    ljh_path = pathlib.Path(ljh_path)  # Convert to Path if it's a string
    base_name = ljh_path.name.split('_chan')[0]
    new_file_name = f"{base_name}_experiment_state.txt"
    return ljh_path.parent / new_file_name


def external_trigger_bin_path_from_ljh_path(ljh_path: Union[str, pathlib.Path]) -> pathlib.Path:
    ljh_path = pathlib.Path(ljh_path)  # Convert to Path if it's a string
    base_name = ljh_path.name.split('_chan')[0]
    new_file_name = f"{base_name}_external_trigger.bin"
    return ljh_path.parent / new_file_name
