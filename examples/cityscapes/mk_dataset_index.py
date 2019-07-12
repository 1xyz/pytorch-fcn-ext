"""Make dataset index files
Usage:
  mk_dataset_index.py cityscape  [--input-folder=<if>] [--output-folder=<of>]
  mk_dataset_index.py -h | --help
Options:
  -h --help             Show this screen.
  --input-folder=<if>   Input folder [default: ~/data/datasets].
  --output-folder=<if>  Output folder [default: ~/data/datasets].
"""

import os
import os.path as osp
from itertools import product
from pathlib import Path
from typing import Iterable, Iterator

from docopt import docopt
from tabulate import tabulate


def walk_files(root: str, *suffixes: Iterable[str]) -> Iterator[str]:
    """
    Generate file by  Walk through the provided folder

    :param root:  A Root folder path
    :param suffixes:  Extra folder arguments
    :return:  A generator of file paths
    """
    prefix_len = len(root)
    root = osp.join(root, *suffixes)
    for root, _, files in os.walk(root):
        root_suffix = root[prefix_len:]
        for f in files:
            yield osp.join(root_suffix, f)


def write_cs_index(input_root: str, output_root: str) -> None:
    """
    Write the test/train/val dataset index for the cityscape dataset to files

    :param input_root: The root directory where the input cityscapes data is hosted
    :param output_root: The root directory where the data is posted
    """
    output_root = osp.join(osp.expanduser(output_root), "Cityscapes", "Cityscapes")
    cityscape_root = osp.join(osp.expanduser(input_root), "Cityscapes", "Cityscapes")

    print(tabulate([["Input_folder", f"{cityscape_root}"],
                    ["Output_Folder", f"{output_root}"]], headers=["Arg", "Value"]))
    print()
    file_counts = []
    raw_file_prefix = "leftImg8bit"
    file_types = ["train", "test", "val"]
    for file_type in file_types:
        key = f"Cityscapes_{raw_file_prefix}_{file_type}.txt"
        cur_file_count = 0
        file_name: str = osp.join(output_root, key)
        with open(file_name, "w") as f:
            for name in walk_files(cityscape_root, raw_file_prefix, file_type):
                f.write(f"{name}\n")
                cur_file_count += 1
        file_counts.append([key, cur_file_count])

    label_file_prefix = "gtFine"
    label_ids = ["color", "labelids"]
    for file_type, label_id in product(file_types, label_ids):
        key = f"Cityscapes_{label_file_prefix}_{file_type}_{label_id}.txt"
        cur_file_count = 0
        file_name: str = osp.join(output_root, key)
        with open(file_name, "w") as f:
            for name in walk_files(cityscape_root, label_file_prefix, file_type):
                base_name: str = Path(name).stem
                if base_name.lower().endswith(label_id):
                    f.write(f"{name}\n")
                    cur_file_count += 1
        file_counts.append([key, cur_file_count])

    print(tabulate(file_counts, headers=["Dataset", "Count"]))


if __name__ == '__main__':
    args = docopt(__doc__, version='1.0')
    if args["cityscape"]:
        write_cs_index(args["--input-folder"], args["--output-folder"])
