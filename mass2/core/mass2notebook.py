"""
Create a starter Marimo notebook for mass2 analysis, based on a template.
"""

import argparse
import subprocess
import os
import pathlib
import importlib.resources as pkg_resources

TEMPLATE_NAME = "notebook_starter_template.py"


def create_notebook(path: pathlib.Path, notebook_path_str: str | None, overwrite: bool) -> None:
    """Create a notebook file from a template, finding data from the data directory."""
    if path.is_file():
        data_dir = path.parent
    elif path.is_dir():
        data_dir = path
    else:
        raise ValueError(f"{path} must point to a file or dir")
    if notebook_path_str is None:
        notebook_path = data_dir/"notebook_from_template.py"
    else:
        notebook_path = pathlib.Path(notebook_path_str)
    if notebook_path.is_file() and not overwrite:
        raise OSError(f"output notebook {notebook_path} exists, pass `-f` if you want to overwrite")
    template_path = pkg_resources.files("mass2").joinpath("data", TEMPLATE_NAME)
    with open(template_path, "r", encoding="utf-8") as fp:
        template_text = fp.read()
        output_text = template_text.replace("TEMPLATE_DIRECTORY", f"r'{data_dir}'")
    with open(notebook_path, "w", encoding="utf-8") as fp:
        fp.write(output_text)
    return notebook_path


def main():
    """Parse command-line arguments and create a Marimo notebook."""
    parser = argparse.ArgumentParser(
        description="Start a mass2 notebook session by loading a set of LJH files",
    )
    parser.add_argument("path", type=str, nargs="?", default=".", help="directory to find LJH files (default: current directory)")
    parser.add_argument("-n", "--notebook-path", type=str, required=False, default=None, help="path to the marimo notebook that will be created")
    parser.add_argument("-f", "--force", action="store_true", help="overwrite existing notebook")
    args = parser.parse_args()
    path = pathlib.Path(args.path).absolute()
    notebook_path = create_notebook(path=path, 
                                    notebook_path_str=args.notebook_path, 
                                    overwrite = args.force)
    print(f"Created file {notebook_path}")
    print(f"Next time:    marimo edit {notebook_path}")
    subprocess.run(["marimo", "edit", notebook_path], check=False, capture_output=True)
