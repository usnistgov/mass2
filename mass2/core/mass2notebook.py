"""
Create a starter Marimo notebook for mass2 analysis, based on a template.
"""

import argparse
import subprocess
import os
import pathlib
import importlib.resources as pkg_resources

TEMPLATE_NAME = "notebook_starter_template.py"


def create_notebook(data_dir: str, notebook_name: str) -> None:
    """Create a notebook file from a template, finding data from the data directory."""
    if os.path.exists(notebook_name):
        raise OSError(f"output notebook {notebook_name} exists")
    template_path = pkg_resources.files("mass2").joinpath("data", TEMPLATE_NAME)
    with open(template_path, "r", encoding="utf-8") as fp:
        template_text = fp.read()
        output_text = template_text.replace("TEMPLATE_DIRECTORY", f"'{data_dir}'")
    with open(notebook_name, "w", encoding="utf-8") as fp:
        fp.write(output_text)


def main():
    """Parse command-line arguments and create a Marimo notebook."""
    parser = argparse.ArgumentParser(
        description="Start a mass2 notebook session by loading a set of LJH files",
    )
    parser.add_argument("data_dir", type=str, nargs="?", default=".", help="directory to find LJH files (default: current directory)")
    parser.add_argument("notebook_name", type=str, help="path to the marimo notebook that will be created")
    args = parser.parse_args()
    data_dir = pathlib.Path(args.data_dir).absolute()
    create_notebook(data_dir=data_dir, notebook_name=args.notebook_name)
    print(f"Created file {args.notebook_name}")
    print(f"Next time:    marimo edit {args.notebook_name}")
    subprocess.run(["marimo", "edit", args.notebook_name], check=False, capture_output=True)
