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
from typing import Iterable, Iterator

from docopt import docopt
from tabulate import tabulate


def walk_files(root: str, *suffixes: Iterable[str]) -> Iterator[str]:
    """
    Generate file by  Walk through the provided folder

    :param root:  A Root folder path
    :return:  A generator of file paths
    """
    root = osp.join(root, *suffixes)
    for _, _, files in os.walk(root):
        for f in files:
            yield f


def write_cs_index(input_root: str, output_root: str) -> None:
    """
    Write the test/train/val dataset index for the cityscape dataset to files

    :param input_root: The root directory where the input cityscapes data is hosted
    :param output_root: The root directory where the data is posted
    """
    output_root = osp.join(osp.expanduser(output_root), "CityScapes", "CityScapes")
    input_root = osp.join(osp.expanduser(input_root), "CityScapes", "CityScapes")
    print(f"i: {input_root}, o: {output_root}")

    file_counts = []
    file_types = ["train", "test", "val"]
    for file_type in file_types:
        n = 0
        file_name: str = osp.join(output_root, f"CityScapes_{file_type}.txt")
        with open(file_name, "w") as f:
            for name in walk_files(input_root, "leftImg8bit", file_type):
                # name looks like strasbourg_000001_058373_leftImg8bit.png
                f.write(f"{name.rpartition('_')[0]}\n")
                n += 1
        file_counts.append([file_name, n])

    print(tabulate(file_counts, headers=["Dataset", "Count"]))


if __name__ == '__main__':
    args = docopt(__doc__, version='1.0')
    if args["cityscape"]:
        write_cs_index(args["--input-folder"], args["--output-folder"])
