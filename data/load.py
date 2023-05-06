"""
Module for loading data files into data structures or into generator functions that return an iterator.
"""

from pprint import pprint
from typing import Callable, Iterator, Tuple, Any, Dict
import numpy as np

import pyBigWig as bw
from BCBio import GFF
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord


def read_sequence(fasta_file: str) -> dict:
    """Read first sequence from FASTA file."""
    return SeqIO.to_dict(SeqIO.parse(open(fasta_file), 'fasta'))


def examine_annotation(gff_file: str):
    """Print GFF annotation file structure."""
    examiner = GFF.GFFExaminer()
    in_handle = open(gff_file)
    print('Parent-Child map:')
    pprint(examiner.parent_child_map(in_handle))
    in_handle.close()
    in_handle = open(gff_file)
    print('GFF tags:')
    pprint(examiner.available_limits(in_handle))
    return


def read_annotation_generator(gff_file: str, seq_dict: dict = None) -> Callable[[], Iterator[SeqRecord]]:
    """
    Read GFF annotation file, returning a generator function of the SeqRecords.
    The returned generator function itself returns a fresh SeqRecords Iterator each time it is called.
    This gets around the issue that a SeqRecords Iterator.
    """
    def seq_records_gen():
        in_handle = open(gff_file)
        seq_records = GFF.parse(in_handle, base_dict=seq_dict)
        return seq_records
    return seq_records_gen


def read_gerp_scorer(gerp_file: str) -> Callable[[str, int, int], np.array]:
    """
    Return the value function which itself returns GERP values from a given BigWig file.
    """
    gerp_bw = bw.open(gerp_file)

    def scorer(seq: str, start: int, end:int):
        return np.array(gerp_bw.values(seq, start, end))
    return scorer
