"""A data translation layer for handling data read/write to files

CSV Output Format:
<character>,<endpoints_list>,<jointpoints_list>,<shortest_edge_length>,<average_edge_length>,<longest_edge_length>
"""
import csv
from typing import List

from character import Character


def write_chars_to_file(chars: List[Character]):
    with open("test/output/text.csv", "w+") as file:
        writer = csv.writer(file)
        for char in chars:
            if not char.usable:
                continue

            writer.writerow(
                [char.letter,
                 char.endpoints,
                 char.jointpoints,
                 char.min_edge_length,
                 char.average_edge_length,
                 char.max_edge_length])
