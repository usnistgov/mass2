import os
import re
from typing import List, Iterator, Tuple

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

def find_ljh_files(folder: str, ext:str = ".ljh") -> List[str]:
    """
    Finds all .ljh files in the given folder and its subfolders.

    Args:
    - folder (str): The root directory to start the search from.

    Returns:
    - List[str]: A list of paths to .ljh files.
    """
    ljh_files = []
    for dirpath, _, filenames in os.walk(folder):
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


    files1_by_channel = {extract_channel_number(f): f for f in files1}
    files2_by_channel = {extract_channel_number(f): f for f in files2}

    matching_pairs = []
    for channel in files1_by_channel:
        if channel in files2_by_channel:
            matching_pairs.append((files1_by_channel[channel], files2_by_channel[channel]))
    # print(f"in match_files_by_channel found {len(matching_pairs)} channel pairs, {limit=}")
    matching_pairs_limited = matching_pairs[:limit]
    # print(f"in match_files_by_channel found {len(matching_pairs)=} after limit of {limit=}")
    return matching_pairs_limited

def experiment_state_path_from_ljh_path(ljh_path):
    # Split the path into directory and filename
    dir_name, file_name = os.path.split(ljh_path)
    
    # Split the filename into parts based on '_chan' and '.ljh'
    base_name, _ = file_name.split('_chan')
    # Create the new filename
    new_file_name = f"{base_name}_experiment_state.txt"
    
    # Join the directory and new filename to form the new path
    experiment_state_path = os.path.join(dir_name, new_file_name)
    
    return experiment_state_path
